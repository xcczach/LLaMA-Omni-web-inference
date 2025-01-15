import os
import subprocess
import requests


def _download_file(url, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    filename = os.path.join(dest_folder, url.split("/")[-1])
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"下载完成: {filename}")
    else:
        print(f"下载失败: {url}")


def install_requirements():
    if os.path.exists("installation_complete"):
        return

    subprocess.run(["pip", "install", "pip==24.0"], check=True)

    subprocess.run(["pip", "install", "-e", "."], check=True)

    if not os.path.exists("fairseq"):
        subprocess.run(
            ["git", "clone", "https://github.com/One-sixth/fairseq.git"], check=True
        )
    os.chdir("fairseq")
    subprocess.run(["pip", "install", "-e", ".", "--no-build-isolation"], check=True)
    os.chdir("..")

    subprocess.run(["pip", "install", "flash-attn", "--no-build-isolation"], check=True)

    import whisper

    whisper.load_model("large-v3", download_root="models/speech_encoder/")

    urls = [
        "https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000",
        "https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json",
    ]
    dest_folder = "vocoder/"
    for url in urls:
        _download_file(url, dest_folder)

    with open("installation_complete", "w") as f:
        f.write("Installation complete.")


if __name__ == "__main__":
    install_requirements()
