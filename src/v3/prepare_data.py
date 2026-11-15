"""Freeze new v3 calibration, validation and untouched held-out inputs."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
os.environ.update(
    HF_HOME=str(BASE / "cache/huggingface"),
    HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
    HF_HUB_DISABLE_PROGRESS_BARS="1",
    TOKENIZERS_PARALLELISM="false",
)


def encode_blocks(tokenizer, texts, count, seed):
    ids = tokenizer.encode("\n\n".join(texts), add_special_tokens=False)
    starts = np.arange(0, len(ids) - 256, 257)
    np.random.default_rng(seed).shuffle(starts)
    if len(starts) < count:
        raise RuntimeError(f"Need {count} token blocks; have {len(starts)}")
    return np.array([ids[int(i):int(i) + 257] for i in starts[:count]], dtype=np.int32)


def main():
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    if (ROOT / "data_manifest.json").exists():
        print("PASS — v3 data manifest already frozen")
        return
    v2 = json.loads((BASE / "v2/inputs.json").read_text())
    imdb_revision = HfApi().dataset_info("stanfordnlp/imdb", token=False).sha
    news = load_dataset("fancyzhx/ag_news", revision=v2["news_revision"])
    shuffled_news = news["train"].shuffle(seed=20261018)
    imdb = load_dataset("stanfordnlp/imdb", revision=imdb_revision)
    shuffled_imdb_train = imdb["train"].shuffle(seed=20261019)
    shuffled_imdb_test = imdb["test"].shuffle(seed=20261019)
    partitions = {
        "news_calibration": shuffled_news[10000:15000]["text"],
        "news_validation": shuffled_news[15000:20000]["text"],
        "news_test": shuffled_news[20000:25000]["text"],
        "imdb_calibration": shuffled_imdb_train[:5000]["text"],
        "imdb_validation": shuffled_imdb_train[5000:10000]["text"],
        "imdb_test": shuffled_imdb_test[:5000]["text"],
    }
    prior_arc = json.loads((BASE / "data/arc_easy.json").read_text())
    prior_arc += json.loads((BASE / "v2/data/arc_easy.json").read_text())
    prior_ids = {row["id"] for row in prior_arc}
    arc = load_dataset(
        "allenai/ai2_arc", "ARC-Challenge", split="test", revision=v2["arc_revision"]
    ).filter(lambda row: row["id"] not in prior_ids)
    arc = arc.shuffle(seed=20261019).select(range(min(400, len(arc))))
    data_root = ROOT / "data"
    data_root.mkdir(exist_ok=True)
    (data_root / "arc_challenge.json").write_text(json.dumps(list(arc)))
    model_paths = {
        "qwen3_1_7b": v2["buddy_model_path"],
        "qwen3_5_2b": v2["model_path"],
    }
    counts = {"calibration": 64, "validation": 64, "test": 128}
    hashes = {}
    for name, model_path in model_paths.items():
        folder = data_root / name
        folder.mkdir(exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        arrays = {}
        for domain in ("news", "imdb"):
            for split, count in counts.items():
                key = f"{domain}_{split}"
                array = encode_blocks(tokenizer, partitions[key], count, 20261020)
                np.save(folder / f"{key}_tokens.npy", array)
                arrays[key] = array
                hashes[f"{name}/{key}"] = hashlib.sha256(array.tobytes()).hexdigest()
        combined = np.stack([row for pair in zip(arrays["news_calibration"], arrays["imdb_calibration"])
                             for row in pair])
        np.save(folder / "calibration_tokens.npy", combined)
        np.save(folder / "train_tokens.npy", combined[:32])
        validation = np.stack([row for pair in zip(arrays["news_validation"], arrays["imdb_validation"])
                               for row in pair])
        np.save(folder / "validation_tokens.npy", validation)
    manifest = {
        "status": "frozen_inputs",
        "models": model_paths,
        "ag_news_revision": v2["news_revision"],
        "imdb_revision": imdb_revision,
        "arc_revision": v2["arc_revision"],
        "arc_challenge_count": len(arc),
        "prior_arc_ids_excluded": len(prior_ids),
        "text_partition_rule": "AG News shuffled train seed 20261018: indices 10000:15000 calibration, 15000:20000 validation, 20000:25000 v3 held-out; v2 only used earlier training indices and official test. IMDB shuffled train seed 20261019: first 5000 calibration, next 5000 validation; official test first 5000 v3 held-out.",
        "block_tokens": 257,
        "blocks_per_domain": counts,
        "array_sha256": hashes,
    }
    (ROOT / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("PASS — v3 disjoint inputs and new held-out sets frozen")


if __name__ == "__main__":
    main()
