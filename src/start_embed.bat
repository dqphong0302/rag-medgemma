@echo off
call conda activate medgemma_env
python -m llama_cpp.server --model D:\medgemma\models\nomic-embed-text-v1.5.Q4_K_M.gguf --port 8001 --embedding --n_ctx 2048 --host 0.0.0.0
pause
