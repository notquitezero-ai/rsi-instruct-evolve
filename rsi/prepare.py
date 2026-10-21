"""One-time setup: download the base model into models/ and build data splits under data/.

    python -m rsi.prepare [--model Qwen/Qwen3-1.7B-Base]
"""
import argparse

from huggingface_hub import snapshot_download

from . import data as D
from .paths import MODELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    a = ap.parse_args()
    local = MODELS / a.model.split("/")[-1]
    snapshot_download(a.model, local_dir=str(local), allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model",
                                                                      "*.jinja", "LICENSE*"])
    print("model ->", local)
    print("data  ->", D.prepare())


if __name__ == "__main__":
    main()
