# Copyright 2025 ByteDance and/or its affiliates.
# ... (license headers) ...

import multiprocessing as mp
import torch
import os
from functools import partial
import gradio as gr
import traceback
from tts.infer_cli import MegaTTS3DiTInfer, convert_to_wav, cut_wav

import uvicorn
from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel

# --- SOLUTION: Make Queues Global ---
# 将队列定义在模块的顶层，使其成为全局变量
# 这是在多进程应用中安全共享状态的标准模式
mp.set_start_method('spawn', force=True)
input_queue = mp.Queue()
output_queue = mp.Queue()
# --- END SOLUTION ---


def model_worker(worker_input_queue, worker_output_queue, device_id):
    # 函数签名保持不变，以清晰地表明它接收队列
    device = None
    if device_id is not None:
        device = torch.device(f'cuda:{device_id}')
    infer_pipe = MegaTTS3DiTInfer(device=device)

    while True:
        task = worker_input_queue.get()
        inp_audio_path, inp_npy_path, inp_text, infer_timestep, p_w, t_w = task
        try:
            convert_to_wav(inp_audio_path)
            wav_path = os.path.splitext(inp_audio_path)[0] + '.wav'
            cut_wav(wav_path, max_len=28)
            with open(wav_path, 'rb') as file:
                file_content = file.read()
            resource_context = infer_pipe.preprocess(file_content, latent_file=inp_npy_path)
            wav_bytes = infer_pipe.forward(resource_context, inp_text, time_step=infer_timestep, p_w=p_w, t_w=t_w)
            worker_output_queue.put(wav_bytes)
        except Exception as e:
            traceback.print_exc()
            print(task, str(e))
            worker_output_queue.put(None)


def main(inp_audio, inp_npy, inp_text, infer_timestep, p_w, t_w):
    # --- SOLUTION: Use Global Queues ---
    # 不再从参数接收队列，直接使用全局变量
    print("Push task to the inp queue |", inp_audio, inp_npy, inp_text, infer_timestep, p_w, t_w)
    input_queue.put((inp_audio, inp_npy, inp_text, infer_timestep, p_w, t_w))
    res = output_queue.get()
    if res is not None:
        return res
    else:
        print("")
        return None


class N8nPayload(BaseModel):
    filename: str
    text: str
    infer_timestep: int = 32
    intelligibility_weight: float = 1.4
    similarity_weight: float = 3.0


def n8n_predict(payload: N8nPayload):
    # --- SOLUTION: Use Global Queues ---
    # 同样，直接使用全局变量
    print(f"Received API request for filename: {payload.filename}")
    base_path = 'assets'
    wav_path = os.path.join(base_path, f"{payload.filename}.wav")
    npy_path = os.path.join(base_path, f"{payload.filename}.npy")

    if not os.path.exists(wav_path) or not os.path.exists(npy_path):
        print(f"Error: Files not found. Searched for {wav_path} and {npy_path}")
        raise HTTPException(status_code=404, detail=f"Files for '{payload.filename}' not found in '{base_path}' directory.")

    task = (
        wav_path, npy_path, payload.text, payload.infer_timestep,
        payload.intelligibility_weight, payload.similarity_weight
    )
    input_queue.put(task)
    result_bytes = output_queue.get()

    if result_bytes:
        return Response(content=result_bytes, media_type="audio/wav")
    else:
        raise HTTPException(status_code=500, detail="Model worker failed to process the request.")


if __name__ == '__main__':
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    devices = devices.split(",") if devices != '' else None
    
    num_workers = 1
    processes = []

    print("Start open workers")
    for i in range(num_workers):
        device_id = i % len(devices) if devices is not None else None
        # 创建进程时，仍然需要将全局队列作为参数传递，这是正确的做法
        p = mp.Process(target=model_worker, args=(input_queue, output_queue, device_id))
        p.start()
        processes.append(p)

    app = FastAPI()

    # --- SOLUTION: Remove Partial ---
    # 直接注册函数，不再需要 partial 来注入队列
    app.post("/api/n8n_predict")(n8n_predict)

    api_interface = gr.Interface(
        fn=main, # 直接使用 main 函数，不再需要 partial
        inputs=[
            gr.Audio(type="filepath", label="Upload .wav"),
            gr.File(type="filepath", label="Upload .npy"),
            "text",
            gr.Number(label="infer timestep", value=32),
            gr.Number(label="Intelligibility Weight", value=1.4),
            gr.Number(label="Similarity Weight", value=3.0)
        ],
        outputs=[gr.Audio(label="Synthesized Audio")],
        title="MegaTTS3",
        description="...",
        concurrency_limit=1
    )

    app = gr.mount_gradio_app(app, api_interface, path="/")
    print("Gradio UI is mounted on '/'")
    print("Custom API endpoint '/api/n8n_predict' is available.")

    uvicorn.run(app, host="0.0.0.0", port=7929)

    print("Shutting down workers...")
    for p in processes:
        p.terminate()
        p.join()