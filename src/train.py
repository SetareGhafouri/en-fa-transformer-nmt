"""Train the English->Persian Transformer.

    python src/train.py --config configs/base.yaml --out checkpoints/base
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nmt.data import Vocab, fasttext_matrix, length_cutoff, load_pairs, tensorise  # noqa: E402
from nmt.model import build_transformer, masked_accuracy, masked_loss  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    ap.add_argument("--out", type=Path, default=Path("checkpoints/base"))
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    d, m, t = cfg["data"], cfg["model"], cfg["train"]
    args.out.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(d["seed"])

    splits = load_pairs(d["fraction"], d["seed"])
    (args.out / "test_pairs.jsonl").write_text(
        "\n".join(json.dumps({"en": s, "fa": f}, ensure_ascii=False) for s, f in splits["test"]))
    src_sents, tgt_sents = zip(*splits["train"])
    src_v = Vocab.build(src_sents, d["vocab_size"], d["min_freq"])
    tgt_v = Vocab.build(tgt_sents, d["vocab_size"], d["min_freq"])
    src_len = length_cutoff(src_sents, d["length_percentile"])
    tgt_len = length_cutoff(tgt_sents, d["length_percentile"])
    src_v.save(args.out / "vocab_en.json")
    tgt_v.save(args.out / "vocab_fa.json")

    src_emb, cov_en = fasttext_matrix(cfg["fasttext"]["en"], src_v, m["embed_dim"])
    tgt_emb, cov_fa = fasttext_matrix(cfg["fasttext"]["fa"], tgt_v, m["embed_dim"])
    meta = {"src_len": src_len, "tgt_len": tgt_len, "fasttext_coverage_en": round(cov_en, 4),
            "fasttext_coverage_fa": round(cov_fa, 4), **{k: len(v) for k, v in splits.items()}}
    print(meta)

    model = build_transformer(len(src_v), len(tgt_v), src_len, tgt_len + 1, dim=m["embed_dim"],
                              heads=m["num_heads"], ff=m["ff_dim"], layers_=m["num_layers"],
                              dropout=m["dropout"], src_emb=src_emb, tgt_emb=tgt_emb,
                              trainable_emb=cfg["fasttext"]["trainable"])
    model.compile(tf.keras.optimizers.Adam(t["learning_rate"]), loss=masked_loss(), metrics=[masked_accuracy()])
    model.summary()

    def dataset(name, shuffle):
        enc, din, dout = tensorise(splits[name], src_v, tgt_v, src_len, tgt_len + 1)
        ds = tf.data.Dataset.from_tensor_slices(((enc, din), dout))
        if shuffle:
            ds = ds.shuffle(10_000, seed=d["seed"])
        return ds.batch(t["batch_size"]).prefetch(tf.data.AUTOTUNE)

    history = model.fit(
        dataset("train", True), validation_data=dataset("val", False), epochs=t["epochs"],
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(str(args.out / "best.weights.h5"), save_best_only=True,
                                               save_weights_only=True),
            tf.keras.callbacks.EarlyStopping(patience=t["patience"], restore_best_weights=True),
        ],
    )
    meta["history"] = {k: [round(float(x), 4) for x in v] for k, v in history.history.items()}
    (args.out / "training.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
