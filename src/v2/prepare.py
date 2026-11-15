"""Revision-pinned fresh inputs for the expanded experiment."""
import json
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
os.environ.update(HF_HOME=str(BASE / "cache/huggingface"), HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
                  HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false")

def main():
    from huggingface_hub import HfApi, snapshot_download
    from datasets import load_dataset
    import numpy as np
    from transformers import AutoTokenizer
    api = HfApi()
    old = json.loads((BASE / "inputs.json").read_text())
    news_revision = api.dataset_info("fancyzhx/ag_news", token=False).sha
    news = load_dataset("fancyzhx/ag_news", revision=news_revision)
    shuffled = news["train"].shuffle(seed=20261018)
    partitions = {"news_calibration": "\n\n".join(shuffled[:5000]["text"]),
                  "validation": "\n\n".join(shuffled[5000:10000]["text"]),
                  "test": "\n\n".join(news["test"].shuffle(seed=20261018)["text"]),
                  "wiki_calibration": (BASE / "data/train.txt").read_text()}
    old_ids = {x["id"] for x in json.loads((BASE / "data/arc_easy.json").read_text())}
    arc = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test", revision=old["arc_revision"])
    arc = arc.filter(lambda x: x["id"] not in old_ids).shuffle(seed=20261018).select(range(400))
    buddy_id = "Qwen/Qwen3-1.7B"
    buddy_rev = api.model_info(buddy_id, token=False).sha
    buddy_path = snapshot_download(buddy_id, revision=buddy_rev, token=False,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "LICENSE*", "README.md"])
    for folder, model_path in [(ROOT, old["model_path"]), (ROOT / "awq_control", buddy_path)]:
        data = folder / "data"
        data.mkdir(parents=True, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        blocks = {}
        for name, text in partitions.items():
            ids = tokenizer.encode(text, add_special_tokens=False)
            starts = np.arange(0, len(ids) - 512, 513)
            np.random.default_rng(20261018).shuffle(starts)
            count = {"news_calibration": 64, "wiki_calibration": 64, "validation": 128, "test": 256}[name]
            if len(starts) < count:
                raise RuntimeError(f"Insufficient tokens: {name}")
            blocks[name] = np.array([ids[i:i + 513] for i in starts[:count]], dtype=np.int32)
        calibration = np.stack([row for pair in zip(blocks["wiki_calibration"], blocks["news_calibration"]) for row in pair])
        np.save(data / "calibration_tokens.npy", calibration)
        np.save(data / "train_tokens.npy", calibration[:32])
        np.save(data / "validation_tokens.npy", blocks["validation"])
        np.save(data / "test_tokens.npy", blocks["test"])
        (data / "arc_easy.json").write_text(json.dumps(list(arc)))
    manifest = {**old, "news_revision": news_revision, "buddy_id": buddy_id, "buddy_revision": buddy_rev,
                "buddy_model_path": buddy_path, "fresh_arc_count": 400, "old_arc_excluded": len(old_ids),
                "partition_rule": "AG News shuffled train records 0:5000 calibration, 5000:10000 validation; official test split held out"}
    (ROOT / "inputs.json").write_text(json.dumps(manifest, indent=2))
    print("PASS — fresh disjoint text and 400 unseen task questions prepared for both models.")

if __name__ == "__main__":
    main()
