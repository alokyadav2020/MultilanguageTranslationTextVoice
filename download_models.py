#!/usr/bin/env python
"""
download_models.py  •  15 Aug 2025

Download and manage the four OPUS-MT “tc-big” checkpoints still public
on Hugging Face:

    en→fr   en→ar   fr→en   ar→en
---------------------------------------------------------------------
Actions
  download   : fetch one pair (--src --tgt) or all four
  list       : show what is already on disk
  cleanup    : delete everything in ./artifacts/models
Examples
  python download_models.py download
  python download_models.py download --src en --tgt fr
  python download_models.py list
  python download_models.py cleanup
"""
from pathlib import Path
import argparse
import logging
import os
import shutil
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

try:
    import torch
    _HAS_TORCH = True
except ModuleNotFoundError:
    torch = None
    _HAS_TORCH = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Current public model IDs
MODEL_MAP = {
    ("en", "fr"): "Helsinki-NLP/opus-mt-tc-big-en-fr",
    ("fr", "en"): "Helsinki-NLP/opus-mt-tc-big-fr-en",
    ("en", "ar"): "Helsinki-NLP/opus-mt-tc-big-en-ar",
    ("ar", "en"): "Helsinki-NLP/opus-mt-tc-big-ar-en",
}

BASE_DIR = Path(__file__).resolve().parent / "artifacts" / "models"
HF_TOKEN = os.getenv("HF_TOKEN")  # optional—falls back to cached login


def download_pair(src: str, tgt: str) -> bool:
    """Download one model direction."""
    repo = MODEL_MAP.get((src, tgt))
    if repo is None:
        log.warning("no public model for %s→%s – skipping", src, tgt)
        return False

    out_dir = BASE_DIR / f"{src}_to_{tgt}"
    if (out_dir / "pytorch_model.bin").exists():
        log.info("%s→%s already present in %s", src, tgt, out_dir)
        return True

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        log.info("Downloading %s → %s", repo, out_dir)
        tok = AutoTokenizer.from_pretrained(repo, use_auth_token=HF_TOKEN)
        model = AutoModelForSeq2SeqLM.from_pretrained(repo,
                                                      use_auth_token=HF_TOKEN)
        tok.save_pretrained(out_dir)
        model.save_pretrained(out_dir)

        if _HAS_TORCH:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            log.info("model moved to %s", device)

        log.info("✅ done %s→%s", src, tgt)
        return True
    except Exception as err:
        log.error("❌ failed %s→%s: %s", src, tgt, err)
        shutil.rmtree(out_dir, ignore_errors=True)
        return False


def list_pairs() -> None:
    for (src, tgt), repo in MODEL_MAP.items():
        d = BASE_DIR / f"{src}_to_{tgt}"
        ok = d.exists() and (d / "pytorch_model.bin").exists()
        status = "✅" if ok else "❌"
        print(f"{status} {src}->{tgt} ({repo}) in {d}")


def cleanup() -> None:
    if BASE_DIR.exists():
        shutil.rmtree(BASE_DIR)
        log.info("removed %s", BASE_DIR)
    else:
        log.info("nothing to clean")


def main() -> None:
    ap = argparse.ArgumentParser(description="OPUS-MT model downloader")
    ap.add_argument("action", choices=["download", "list", "cleanup"])
    ap.add_argument("--src", choices=["en", "fr", "ar"])
    ap.add_argument("--tgt", choices=["en", "fr", "ar"])
    args = ap.parse_args()

    if args.action == "cleanup":
        cleanup()
        return
    if args.action == "list":
        list_pairs()
        return

    targets = [(args.src, args.tgt)] if args.src and args.tgt else MODEL_MAP
    for src, tgt in targets:
        download_pair(src, tgt)


if __name__ == "__main__":
    main()
