from huggingface_hub import hf_hub_download
import os

local_dir = "models"

# 1. MedGemma
repo_id = "SandLogicTechnologies/MedGemma-4B-IT-GGUF"
filename = "medgemma-4b-it_Q4_K_M.gguf"
print(f"Downloading {filename} from {repo_id}...")
try:
    path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir, local_dir_use_symlinks=False)
    print(f"Successfully downloaded to: {path}")
except Exception as e:
    print(f"Error downloading MedGemma: {e}")

# 2. Embedding Model
repo_id_embed = "nomic-ai/nomic-embed-text-v1.5-GGUF"
filename_embed = "nomic-embed-text-v1.5.Q4_K_M.gguf"
print(f"Downloading {filename_embed} from {repo_id_embed}...")
try:
    path = hf_hub_download(repo_id=repo_id_embed, filename=filename_embed, local_dir=local_dir, local_dir_use_symlinks=False)
    print(f"Successfully downloaded to: {path}")
except Exception as e:
    print(f"Error downloading embedding: {e}")
