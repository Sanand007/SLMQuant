"""Fresh-test, long-generation MLX-affine comparison using the frozen v3 operators."""

import hashlib
import json
import os
from pathlib import Path
import shutil

V3 = Path(__file__).resolve().parent
BASE = V3.parent
STUDY = V3 / "studies/wide_mlx_affine_qwen3_1_7b"
SOURCE = V3 / "data/qwen3_1_7b"
os.environ.update(
    HF_HOME=str(BASE / "cache/huggingface"),
    HF_HUB_OFFLINE="1",
    HF_DATASETS_OFFLINE="1",
    HF_HUB_DISABLE_PROGRESS_BARS="1",
    TOKENIZERS_PARALLELISM="false",
)

import mlx_affine_study as shared


def prepare():
    import numpy as np
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from prepare_data import encode_blocks

    original = json.loads((V3 / "data_manifest.json").read_text())
    v2_inputs = json.loads((BASE / "v2/inputs.json").read_text())
    folder = STUDY / "data"
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = STUDY / "fresh_data_manifest.json"
    news_path = folder / "news_test_tokens.npy"
    imdb_path = folder / "imdb_test_tokens.npy"
    arc_path = folder / "arc_easy.json"
    if not manifest_path.exists():
        news = load_dataset("fancyzhx/ag_news", split="train", revision=original["ag_news_revision"])
        news = news.shuffle(seed=20261018)
        imdb = load_dataset("stanfordnlp/imdb", split="test", revision=original["imdb_revision"])
        imdb = imdb.shuffle(seed=20261019)
        old_ids = set()
        for path in (BASE / "data/arc_easy.json", BASE / "v2/data/arc_easy.json",
                     V3 / "data/arc_challenge.json"):
            old_ids.update(row["id"] for row in json.loads(path.read_text()))
        arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test",
                           revision=original["arc_revision"])
        arc = arc.filter(lambda row: row["id"] not in old_ids).shuffle(seed=20261021)
        if len(arc) < 400:
            raise RuntimeError(f"Only {len(arc)} untouched ARC-Challenge questions remain")
        arc_rows = list(arc.select(range(400)))
        tokenizer = AutoTokenizer.from_pretrained(v2_inputs["buddy_model_path"], local_files_only=True)
        news_tokens = encode_blocks(tokenizer, news[25000:30000]["text"], 128, 20261022)
        imdb_tokens = encode_blocks(tokenizer, imdb[5000:10000]["text"], 128, 20261023)
        np.save(news_path, news_tokens)
        np.save(imdb_path, imdb_tokens)
        arc_path.write_text(json.dumps(arc_rows))
        manifest = {
            "status": "frozen_before_search",
            "model": v2_inputs["buddy_id"],
            "ag_news_revision": original["ag_news_revision"],
            "imdb_revision": original["imdb_revision"],
            "arc_revision": original["arc_revision"],
            "news_partition": "train shuffled seed 20261018 indices 25000:30000",
            "imdb_partition": "official test shuffled seed 20261019 indices 5000:10000",
            "arc_partition": "test excluding all earlier v1/v2/v3 IDs; shuffled seed 20261021; first 400",
            "arc_count": len(arc_rows),
            "arc_ids_sha256": hashlib.sha256(json.dumps([row["id"] for row in arc_rows]).encode()).hexdigest(),
            "arrays_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in (news_path, imdb_path)},
        }
        shared.write(manifest_path, manifest)
    else:
        manifest = shared.read(manifest_path)
        if manifest["status"] != "frozen_before_search":
            raise RuntimeError("Fresh data manifest is not frozen")
        for path in (news_path, imdb_path):
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["arrays_sha256"][path.name]:
                raise RuntimeError(f"Fresh held-out array changed: {path.name}")
        ids = [row["id"] for row in json.loads(arc_path.read_text())]
        if hashlib.sha256(json.dumps(ids).encode()).hexdigest() != manifest["arc_ids_sha256"]:
            raise RuntimeError("Fresh ARC IDs changed")

    for name in ("calibration", "train", "validation"):
        target = folder / f"{name}_tokens.npy"
        if not target.exists():
            shutil.copyfile(SOURCE / f"{name}_tokens.npy", target)
    test_path = folder / "test_tokens.npy"
    if not test_path.exists():
        news_tokens = np.load(news_path)
        imdb_tokens = np.load(imdb_path)
        np.save(test_path, np.stack([row for pair in zip(news_tokens, imdb_tokens) for row in pair]))
    inputs = {
        "model_id": v2_inputs["buddy_id"],
        "model_path": v2_inputs["buddy_model_path"],
        "model_revision": v2_inputs["buddy_revision"],
        "v3_data_manifest": str(manifest_path),
    }
    if not (STUDY / "inputs.json").exists():
        shared.write(STUDY / "inputs.json", inputs)


def source_hashes():
    files = (
        Path(__file__), Path(shared.__file__), STUDY / "protocol.json",
        STUDY / "fresh_data_manifest.json", V3 / "wide_search_plan.md",
        V3 / "protocol_frozen.md", V3 / "data_manifest.json",
        BASE / "experiment.py", BASE / "search.py", BASE / "v2/runner.py",
        BASE / "v2/operators.py",
    )
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


shared.STUDY = STUDY
shared.prepare = prepare
shared.source_hashes = source_hashes

if __name__ == "__main__":
    shared.main()
