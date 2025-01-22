from download import install_requirements

install_requirements()


import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


from ml_web_inference import (
    expose,
    Request,
    StreamingResponse,
)
import torch
import io
import argparse
import torchaudio
import tempfile
from omni_speech.conversation import conv_templates, SeparatorStyle
from omni_speech.model.builder import load_pretrained_model
from omni_speech.utils import disable_torch_init
from omni_speech.datasets.preprocess import tokenizer_speech_token
from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder
import whisper
import json
import setproctitle

tokenizer = None
model = None


def ctc_postprocess(tokens, blank):
    _toks = tokens.squeeze(0).tolist()
    deduplicated_toks = [v for i, v in enumerate(_toks) if i == 0 or v != _toks[i - 1]]
    hyp = [v for v in deduplicated_toks if v != blank]
    hyp = " ".join(list(map(str, hyp)))
    return hyp


async def inference(request: Request) -> StreamingResponse:
    data = await request.json()
    sample_rate = data["sample_rate"]
    audio_data = data["audio_data"]

    qs = "<speech>\nPlease directly answer the questions in the user's speech."
    input_type = "mel"
    mel_size = 128
    vocoder_cfg = "vocoder/config.json"
    vocoder = "vocoder/g_00500000"

    device = model.device
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        torchaudio.save(
            f, torch.tensor(audio_data).unsqueeze(0), sample_rate, format="wav"
        )
        speech_file = f.name
        speech = whisper.load_audio(speech_file)

    conv = conv_templates["llama_3"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    if input_type == "raw":
        speech_tensor = torch.from_numpy(speech)
        if model.config.speech_normalize:
            speech_tensor = torch.nn.functional.layer_norm(speech, speech.shape)
    elif input_type == "mel":
        speech_tensor = whisper.pad_or_trim(speech)
        speech_tensor = whisper.log_mel_spectrogram(
            speech_tensor, n_mels=mel_size
        ).permute(1, 0)
    speech_length = torch.LongTensor([speech.shape[0]])

    input_ids = tokenizer_speech_token(prompt, tokenizer, return_tensors="pt")

    input_ids = input_ids.to(device).unsqueeze(0)
    speech_tensor = speech_tensor.to(dtype=torch.float16, device=device).unsqueeze(0)
    speech_length = speech_length.to(device).unsqueeze(0)

    print(
        "input_ids shape: ",
        input_ids.shape,
        "speech_tensor shape: ",
        speech_tensor.shape,
        "speech_length shape: ",
        speech_length.shape,
    )

    outputs = model.generate(
        input_ids,
        speech=speech_tensor,
        speech_lengths=speech_length,
        do_sample=False,
        temperature=0,
        top_p=None,
        num_beams=1,
        max_new_tokens=256,
        use_cache=True,
        pad_token_id=128004,
        streaming_unit_gen=False,
    )
    output_ids, output_units = outputs
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    output_units = ctc_postprocess(output_units, blank=model.config.unit_vocab_size)

    with open(vocoder_cfg, "r") as f:
        vocoder_cfg = json.load(f)
    vocoder = CodeHiFiGANVocoder(vocoder, vocoder_cfg)
    vocoder = vocoder.to(device)

    data = [list(map(int, output_units.split()))]
    data = torch.LongTensor(data).to(device).squeeze(0)
    x = {"code": data.view(1, -1)}

    with torch.no_grad():
        waveform = vocoder(x, True)
    result_arr = waveform.squeeze().cpu().numpy().reshape(1, -1)
    result = io.BytesIO()
    torchaudio.save(result, torch.tensor(result_arr), 16000, format="wav")
    result.seek(0)
    return StreamingResponse(result, media_type="application/octet-stream")


def init():
    global tokenizer, model
    model_path = "ICTNLP/Llama-3.1-8B-Omni"
    disable_torch_init()
    tokenizer, model, _ = load_pretrained_model(
        model_path, None, is_lora=False, s2s=True
    )


def hangup():
    global tokenizer, model
    del tokenizer
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    setproctitle.setproctitle("llamaomni-web-inference")
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9234)
    parser.add_argument("--api-name", type=str, default="llamaomni")
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
