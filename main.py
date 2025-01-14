from ml_web_inference import expose, Request, StreamingResponse, get_proper_device, get_model_size_mb
import torch
import io
import argparse
import torchaudio
from inference import OmniInference
import tempfile
import numpy as np

client = None
device = None
model_size_mb = 3500

async def inference(request: Request) -> StreamingResponse:
    data = await request.json()
    sample_rate = data["sample_rate"]
    audio_data = data["audio_data"]
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        torchaudio.save(f, torch.tensor(audio_data).unsqueeze(0), sample_rate, format="wav")
        chunk_generator = client.run_AT_batch_stream(f.name)
        chunks = []
        for chunk in chunk_generator:
            chunk_audio_data = np.frombuffer(chunk, dtype=np.int16)
            chunk_audio_data = chunk_audio_data.reshape(-1, 1)
            chunks.append(chunk_audio_data)

    result_arr = np.concatenate(chunks, axis=0).astype(np.int16)
    result_arr = result_arr.transpose(1, 0)
    result = io.BytesIO()
    torchaudio.save(result, torch.tensor(result_arr), 24000, format="wav")
    result.seek(0)
    return StreamingResponse(result, media_type="application/octet-stream")


def init():
    global client, device
    device = get_proper_device(model_size_mb)
    client = OmniInference('./checkpoint', f"cuda:{device}")
    client.warm_up()

def hangup():
    global client
    del client
    torch.cuda.empty_cache()


if  __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9234)
    parser.add_argument("--api-name", type=str, default="miniomni")
    parser.add_argument("--hangup-timeout-sec", type=int, default=900)
    parser.add_argument("--hangup-interval-sec", type=int, default=60)
    args = parser.parse_args()
    expose(
        args.api_name,
        inference,
        port=args.port,
        hangup_timeout_sec=args.hangup_timeout_sec,
        hangup_interval_sec=args.hangup_interval_sec,
        init_function=init,
        hangup_function=hangup,
    )