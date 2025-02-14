import requests
import soundfile as sf
import io

url = "http://localhost:9234/llamaomni"
audio_data, sample_rate = sf.read("en_sample.wav")
data = {"sample_rate": sample_rate, "audio_data": audio_data.tolist()}

response = requests.post(url, json=data, stream=True)

if response.status_code == 200:
    audio_chunks = []

    buffer = io.BytesIO()
    for chunk in response.iter_content(chunk_size=4096):
        buffer.write(chunk)

    buffer.seek(0)

    audio_data, sample_rate = sf.read(buffer)
    sf.write("output_audio.wav", audio_data, sample_rate)

    print("Audio saved at output_audio.wav")
else:
    print(f"Bad request：{response.status_code}")
