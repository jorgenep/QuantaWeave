import json
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# QuantaWeave-30B is initialized from this checkpoint and keeps its tokenizer.
TOKENIZER_ID = "Qwen/Qwen3.8-27B" 
TOTAL_TOKENS = 100_000_000_000

DATASETS_CONFIG = [
    {
        "name": "Code (75%)",
        # StarCoderData is massive and native Parquet. The "default" config streams
        # every language mixed together; to restrict it, pass data_dir="python" (or
        # "cpp", "java", ...) to load_dataset below.
        "repo": "bigcode/starcoderdata",
        "config": "default", 
        "split": "train",
        "text_column": "content", 
        "target_tokens": int(TOTAL_TOKENS * 0.75),
        "output_file": "data/raw/dataset_code_75B.jsonl"
    },
    {
        "name": "Management, Logic & Docs (25%)",
        # FineWeb-Edu is heavily filtered for educational/structural reasoning
        "repo": "HuggingFaceFW/fineweb-edu",
        "config": "sample-100BT",
        "split": "train",
        "text_column": "text", 
        "target_tokens": int(TOTAL_TOKENS * 0.25),
        "output_file": "data/raw/dataset_mgmt_25B.jsonl"
    }
]

def compile_dataset():
    print(f"Loading tokenizer: {TOKENIZER_ID}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    
    for ds_info in DATASETS_CONFIG:
        print(f"\n--- Processing {ds_info['name']} ---")
        print(f"Target Tokens: {ds_info['target_tokens']:,}")
        
        # Stream the dataset so nothing is downloaded up front
        dataset = load_dataset(
            ds_info["repo"], 
            ds_info["config"], 
            split=ds_info["split"], 
            streaming=True,
        )
        
        current_tokens = 0
        first_row = True
        
        # Open output file (truncates any previous run)
        with open(ds_info["output_file"], 'w', encoding='utf-8') as f:
            with tqdm(total=ds_info["target_tokens"], desc="Tokens Processed", unit="tok") as pbar:
                for row in dataset:
                    # FIX: Debug check on the very first row to ensure the column exists
                    if first_row:
                        if ds_info["text_column"] not in row:
                            raise ValueError(f"Column '{ds_info['text_column']}' not found! Available columns: {list(row.keys())}")
                        first_row = False

                    if current_tokens >= ds_info["target_tokens"]:
                        break
                        
                    text = row.get(ds_info["text_column"])
                    if not text:
                        continue
                        
                    tokens = tokenizer.encode(text, add_special_tokens=False)
                    token_count = len(tokens)
                    
                    if token_count == 0:
                        continue
                    
                    f.write(json.dumps({"text": text}) + '\n')
                    
                    current_tokens += token_count
                    pbar.update(token_count)

        print(f"Completed {ds_info['name']}. Saved to {ds_info['output_file']}")

if __name__ == "__main__":
    compile_dataset()
    print("\nDataset compilation complete. You can now shuffle the resulting JSONL files for MoE training.")