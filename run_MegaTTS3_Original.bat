call conda activate megatts3-env

set CUDA_VISIBLE_DEVICES=0

python tts/gradio_api.py
pause