"""Build and estimate tunable QuantaWeave dense and sparse-MoE architectures."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


PRESETS_PATH = Path(__file__).parent.parent / "configs" / "architecture_presets.json"


@dataclass(frozen=True)
class ModelSpec:
    mode: str
    size: str
    hidden_size: int
    layers: int
    num_experts: int
    top_k: int
    vocab_size: int
    ffn_multiplier: float
    tied_embeddings: bool
    capacity_factor: float = 1.25
    min_expert_capacity: int = 4

    @property
    def ffn_size(self) -> int:
        return round(self.hidden_size * self.ffn_multiplier)

    @property
    def embedding_params(self) -> int:
        multiplier = 1 if self.tied_embeddings else 2
        return multiplier * self.vocab_size * self.hidden_size

    @property
    def attention_params(self) -> int:
        return self.layers * 4 * self.hidden_size**2

    @property
    def expert_params(self) -> int:
        experts = self.num_experts if self.mode == "moe" else 1
        return self.layers * experts * 3 * self.hidden_size * self.ffn_size

    @property
    def active_expert_params(self) -> int:
        experts = self.top_k if self.mode == "moe" else 1
        return self.layers * experts * 3 * self.hidden_size * self.ffn_size

    @property
    def total_params(self) -> int:
        return self.embedding_params + self.attention_params + self.expert_params

    @property
    def active_params(self) -> int:
        return self.embedding_params + self.attention_params + self.active_expert_params

    def manifest(self) -> dict:
        return {
            "model_name": f"QuantaWeave-{self.mode}-{self.size}",
            "model_type": f"quantweave_{self.mode}",
            "mode": self.mode,
            "size": self.size,
            "hidden_size": self.hidden_size,
            "layers": self.layers,
            "ffn_size": self.ffn_size,
            "vocab_size": self.vocab_size,
            "tied_embeddings": self.tied_embeddings,
            "num_experts": self.num_experts if self.mode == "moe" else 1,
            "top_k": self.top_k if self.mode == "moe" else 1,
            "inactive_experts_per_token": max(
                0, (self.num_experts if self.mode == "moe" else 1)
                - (self.top_k if self.mode == "moe" else 1)
            ),
            "active_expert_fraction": (
                (self.top_k / self.num_experts) if self.mode == "moe" else 1.0
            ),
            "capacity_factor": self.capacity_factor,
            "min_expert_capacity": self.min_expert_capacity,
            "estimated_total_params": self.total_params,
            "estimated_active_params": self.active_params,
        }


def load_presets() -> dict:
    return json.loads(PRESETS_PATH.read_text())


def make_spec(mode: str, size: str, **overrides: object) -> ModelSpec:
    presets = load_presets()
    if size not in presets:
        raise ValueError(f"Unknown size '{size}'. Choose from: {', '.join(presets)}")
    if mode not in {"dense", "moe"}:
        raise ValueError("mode must be 'dense' or 'moe'")

    values = {**presets[size], **{key: value for key, value in overrides.items() if value is not None}}
    values["mode"] = mode
    values["size"] = size
    if mode == "dense":
        values["num_experts"] = 1
        values["top_k"] = 1
    if values["top_k"] > values["num_experts"]:
        raise ValueError("active experts (top_k) cannot exceed total experts")
    return ModelSpec(**values)


def billions(value: int) -> float:
    return value / 1_000_000_000


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an QuantaWeave architecture manifest")
    parser.add_argument("--mode", choices=("dense", "moe"), default="moe")
    parser.add_argument("--size", choices=tuple(load_presets()), default="m")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--num-experts", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--total-experts", dest="num_experts", type=int,
                        help="Total expert pools for MoE; alias for --num-experts")
    parser.add_argument("--active-experts", dest="top_k", type=int,
                        help="Experts selected per token; alias for --top-k")
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--ffn-multiplier", type=float)
    parser.add_argument("--tied-embeddings", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--capacity-factor", type=float, default=1.25)
    parser.add_argument("--min-expert-capacity", type=int, default=4)
    args = parser.parse_args()

    try:
        spec = make_spec(
            args.mode,
            args.size,
            hidden_size=args.hidden_size,
            layers=args.layers,
            num_experts=args.num_experts,
            top_k=args.top_k,
            vocab_size=args.vocab_size,
            ffn_multiplier=args.ffn_multiplier,
            tied_embeddings=args.tied_embeddings,
            capacity_factor=args.capacity_factor,
            min_expert_capacity=args.min_expert_capacity,
        )
    except ValueError as error:
        parser.error(str(error))
    manifest = spec.manifest()
    print(
        f"{manifest['model_name']}: total={billions(spec.total_params):.2f}B, "
        f"active/token={billions(spec.active_params):.2f}B"
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
