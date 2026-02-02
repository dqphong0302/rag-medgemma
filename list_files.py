from huggingface_hub import list_repo_files

repo_id = "SandLogicTechnologies/MedGemma-4B-IT-GGUF"
print(f"Listing files in {repo_id}...")
try:
    files = list_repo_files(repo_id)
    for f in files:
        print(f)
except Exception as e:
    print(f"Error listing files: {e}")
