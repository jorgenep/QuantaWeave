import multiprocessing
from pathlib import Path

from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer

# QuantaWeave-30B uses the initialization checkpoint's tokenizer.
TOKENIZER_ID = "Qwen/Qwen3.8-27B"
SEQ_LENGTH = 4096
NUM_WORKERS = multiprocessing.cpu_count()
OUTPUT_DIR = "./data/packed/packed_100B_dataset"
RAW_FILES = ("data/raw/dataset_code_75B.jsonl", "data/raw/dataset_mgmt_25B.jsonl")

def group_texts(examples):
    # Concatenate all token ids in the batch
    concatenated = sum(examples["input_ids"], [])

    # Keep only whole SEQ_LENGTH chunks; the leftover tail of the batch is dropped
    total_length = (len(concatenated) // SEQ_LENGTH) * SEQ_LENGTH
    chunks = [concatenated[i : i + SEQ_LENGTH] for i in range(0, total_length, SEQ_LENGTH)]

    # Axolotl's pre-tokenized format needs exactly input_ids, attention_mask and labels
    return {
        "input_ids": chunks,
        "attention_mask": [[1] * SEQ_LENGTH for _ in chunks],
        "labels": [list(chunk) for chunk in chunks],
    }

def main():
    missing = [path for path in RAW_FILES if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing raw data: {', '.join(missing)}. Run `python3 src/data.py` to compile it first."
        )

    print(f"Loading tokenizer: {TOKENIZER_ID}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError(f"Tokenizer {TOKENIZER_ID} defines no EOS token to separate documents")

    # Load raw JSONL files (Hugging Face maps these via disk, saving RAM)
    print("Mapping raw JSONL files to disk...")
    code_ds = load_dataset("json", data_files="data/raw/dataset_code_75B.jsonl", split="train")
    mgmt_ds = load_dataset("json", data_files="data/raw/dataset_mgmt_25B.jsonl", split="train")

    def tokenize_fn(examples):
        encoded = tokenizer(
            examples["text"],
            truncation=False,
            padding=False,
            return_attention_mask=False,
            add_special_tokens=False,
        )
        # Qwen tokenizers add no special tokens themselves, so append EOS explicitly
        # to mark each document boundary once documents are concatenated and packed.
        return {"input_ids": [ids + [eos_id] for ids in encoded["input_ids"]]}

    # Tokenize independently using all CPU cores
    print("Tokenizing Code Dataset...")
    code_tokenized = code_ds.map(
        tokenize_fn, batched=True, num_proc=NUM_WORKERS, remove_columns=["text"]
    )
    
    print("Tokenizing Management Dataset...")
    mgmt_tokenized = mgmt_ds.map(
        tokenize_fn, batched=True, num_proc=NUM_WORKERS, remove_columns=["text"]
    )

    # Interleave at the exact 75/25 ratio
    print("Interleaving datasets...")
    mixed_ds = interleave_datasets([code_tokenized, mgmt_tokenized], probabilities=[0.75, 0.25])

    # Pack into exact SEQ_LENGTH chunks
    print(f"Packing into {SEQ_LENGTH}-token sequences...")
    packed_ds = mixed_ds.map(
        group_texts, batched=True, num_proc=NUM_WORKERS, remove_columns=["input_ids"]
    )

    # Save to final binary Arrow format
    print(f"Saving ready-to-train dataset to {OUTPUT_DIR}...")
    packed_ds.save_to_disk(OUTPUT_DIR)
    print("Done! Ready for training.")

if __name__ == "__main__":
    main()