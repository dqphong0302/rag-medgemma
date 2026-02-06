import pandas as pd
from pathlib import Path
from huggingface_hub import hf_hub_download
import json

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent / "data"
DATASETS_DIR = DATA_DIR / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

def download_vietnamese_data():
    print("Downloading hungnm/vietnamese-medical-qa...")
    
    # Download parquet file
    file_path = hf_hub_download(
        repo_id="hungnm/vietnamese-medical-qa",
        filename="data/train-00000-of-00001.parquet",
        repo_type="dataset",
        local_dir=DATASETS_DIR
    )
    
    print(f"Downloaded to: {file_path}")
    
    # Read and process
    df = pd.read_parquet(file_path)
    print(f"Loaded {len(df)} records")
    
    # Save as JSON for easier inspection and loading
    output_file = DATASETS_DIR / "vi_medqa.json"
    
    # Convert to list of dicts
    records = df.to_dict('records')
    
    # Save
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
        
    print(f"Saved processed data to: {output_file}")
    
    # Peak at data
    print("\nSample Data:")
    print(json.dumps(records[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    download_vietnamese_data()
