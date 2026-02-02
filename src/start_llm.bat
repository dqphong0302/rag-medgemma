@echo off
call conda activate medgemma_env
python -m llama_cpp.server --model D:\medgemma\models\medgemma-4b-it_Q4_K_M.gguf --port 8000 --n_ctx 4096 --host 0.0.0.0
pause
