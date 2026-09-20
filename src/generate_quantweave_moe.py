"""Generate character-level text from an QuantaWeave MoE checkpoint."""

import argparse
import json
from pathlib import Path

import torch

from quantweave_moe_model import QuantaWeaveConfig, QuantaWeaveMoEForCausalLM
from train_quantweave_moe import select_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/outputs/quantweave-moe-out"))
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", choices=("auto", "cuda", "rocm", "xpu", "cpu"), default="auto")
    args = parser.parse_args()

    config = QuantaWeaveConfig(**json.loads((args.checkpoint / "config.json").read_text()))
    vocab = json.loads((args.checkpoint / "vocab.json").read_text())
    inverse_vocab = {value: key for key, value in vocab.items()}
    unknown_id = vocab["<unk>"]
    device = select_device(args.device)

    model = QuantaWeaveMoEForCausalLM(config).to(device)
    if device.type in {"cuda", "xpu"}:
        model = model.bfloat16()
    checkpoint = torch.load(args.checkpoint / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    prompt_ids = [vocab.get(character, unknown_id) for character in args.prompt]
    generated = prompt_ids[:]
    with torch.inference_mode():
        for _ in range(args.tokens):
            context = torch.tensor([generated[-config.max_sequence_length:]], dtype=torch.long, device=device)
            logits = model(context)["logits"][:, -1, :].float()
            probabilities = (logits / max(args.temperature, 0.01)).softmax(dim=-1)
            next_id = torch.multinomial(probabilities, 1).item()
            generated.append(next_id)
            if next_id == vocab.get("<eos>"):
                break

    text = "".join(inverse_vocab.get(token_id, "?") for token_id in generated)
    print(text)


if __name__ == "__main__":
    main()
