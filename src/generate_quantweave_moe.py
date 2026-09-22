"""Generate text from a QuantaWeave MoE checkpoint (character-level or BPE)."""

import argparse
import json
from pathlib import Path

import torch

from data_pipeline import load_tokenizer
from quantweave_moe_model import QuantaWeaveConfig, QuantaWeaveMoEForCausalLM
from train_quantweave_moe import autocast_context, select_device


def sample_text(model, tokenizer, prompt: str, tokens: int, temperature: float, device: torch.device, precision: str = "auto") -> str:
    """Autoregressive sampling; stops at <eos>. Works for any model returning {"logits": ...}."""
    generated = tokenizer.encode(prompt) or [tokenizer.unk_id]
    context_length = model.config.max_sequence_length - 1        # the last position of a training window is never trained
    with torch.inference_mode():
        for _ in range(tokens):
            context = torch.tensor([generated[-context_length:]], dtype=torch.long, device=device)
            with autocast_context(device, precision):
                logits = model(context)["logits"][:, -1, :].float()
            logits[:, tokenizer.vocab_size:] = float("-inf")        # ids the tokenizer never had are untrained noise
            probabilities = (logits / max(temperature, 0.01)).softmax(dim=-1)
            next_id = torch.multinomial(probabilities, 1).item()
            generated.append(next_id)
            if next_id == tokenizer.eos_id:
                break
    return tokenizer.decode(generated)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/outputs/quantweave-moe-out"))
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--device", choices=("auto", "cuda", "rocm", "xpu", "cpu"), default="auto")
    args = parser.parse_args()

    config = QuantaWeaveConfig(**json.loads((args.checkpoint / "config.json").read_text()))
    tokenizer = load_tokenizer(args.checkpoint)
    device = select_device(args.device)
    model = QuantaWeaveMoEForCausalLM(config).to(device)
    model.load_state_dict(torch.load(args.checkpoint / "model.pt", map_location=device, weights_only=False)["model"])
    model.eval()
    print(sample_text(model, tokenizer, args.prompt, args.tokens, args.temperature, device, args.precision))


if __name__ == "__main__":
    main()
