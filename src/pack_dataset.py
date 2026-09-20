import multiprocessing
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer

# QuantaWeave-30B uses the initialization checkpoint's tokenizer.
TOKENIZER_ID = "Qwen/Qwen3.8-27B"
SEQ_LENGTH = 4096
NUM_WORKERS = multiprocessing.cpu_count()
OUTPUT_DIR = "./data/packed/packed_100B_dataset"

def group_texts(examples):
    # Concatenate all texts in the batch
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    # Drop the small leftover chunk at the end of the batch
    if total_length >= SEQ_LENGTH:
        total_length = (total_length // SEQ_LENGTH) * SEQ_LENGTH
        
    # Split by chunks of SEQ_LENGTH
    result = {
        k: [t[i : i + SEQ_LENGTH] for i in range(0, total_length, SEQ_LENGTH)]
        for k, t in concatenated_examples.items()
    }
    return result

def main():
    print(f"Loading tokenizer: {TOKENIZER_ID}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    
    # Load raw JSONL files (Hugging Face maps these via disk, saving RAM)
    print("Mapping raw JSONL files to disk...")
    code_ds = load_dataset("json", data_files="data/raw/dataset_code_75B.jsonl", split="train")
    mgmt_ds = load_dataset("json", data_files="data/raw/dataset_mgmt_25B.jsonl", split="train")

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=False,
            padding=False,
            return_attention_mask=False,
            add_special_tokens=True # Adds EOS token between documents
        )

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
        group_texts, batched=True, num_proc=NUM_WORKERS
    )

    # Save to final binary Arrow format
    print(f"Saving ready-to-train dataset to {OUTPUT_DIR}...")
    packed_ds.save_to_disk(OUTPUT_DIR)
    print("Done! Ready for training.")

if __name__ == "__main__":
    main()