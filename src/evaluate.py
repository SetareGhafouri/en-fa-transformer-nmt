"""Translate held-out sentences and score them with BLEU, chrF, ROUGE and BERTScore.

References are the *original cleaned Persian sentences*, not sentences reconstructed from the
model's truncated, UNK-replaced target ids. ROUGE uses a whitespace tokenizer: the default
``rouge_score`` tokenizer silently deletes every non-Latin character, which makes it meaningless
for Persian.

    python src/evaluate.py --run checkpoints/base --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import sacrebleu
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nmt.data import EOS, SOS, Vocab, pad  # noqa: E402
from nmt.model import build_transformer, greedy_decode  # noqa: E402


class WhitespaceTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return text.split()


def rouge(refs, hyps) -> dict[str, float]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], tokenizer=WhitespaceTokenizer())
    scores = [scorer.score(r, h) for r, h in zip(refs, hyps)]
    return {k: float(np.mean([s[k].fmeasure for s in scores])) for k in ("rouge1", "rouge2", "rougeL")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--no-bertscore", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    m, e = cfg["model"], cfg["eval"]
    meta = json.loads((args.run / "training.json").read_text())
    src_v, tgt_v = Vocab.load(args.run / "vocab_en.json"), Vocab.load(args.run / "vocab_fa.json")
    src_len, tgt_len = meta["src_len"], meta["tgt_len"] + 1

    model = build_transformer(len(src_v), len(tgt_v), src_len, tgt_len, dim=m["embed_dim"], heads=m["num_heads"],
                              ff=m["ff_dim"], layers_=m["num_layers"], dropout=m["dropout"])
    model.load_weights(str(args.run / "best.weights.h5"))

    pairs = [json.loads(l) for l in (args.run / "test_pairs.jsonl").read_text().splitlines()][: e["max_samples"]]
    srcs, refs = [p["en"] for p in pairs], [p["fa"] for p in pairs]
    hyps = []
    for i in tqdm(range(0, len(srcs), args.batch_size), desc="translating"):
        enc = np.asarray([pad(src_v.encode(s.split()), src_len) for s in srcs[i:i + args.batch_size]], np.int32)
        out = greedy_decode(model, enc, tgt_len, tgt_v.stoi[SOS], tgt_v.stoi[EOS])
        hyps += [tgt_v.decode(row) for row in out]

    results = {
        "n": len(hyps),
        "bleu": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2),
        "chrf": round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2),
        **{k: round(v, 4) for k, v in rouge(refs, hyps).items()},
        "empty_hypotheses": sum(not h.strip() for h in hyps),
    }
    if not args.no_bertscore:
        from bert_score import score

        P, R, F = score(hyps, refs, model_type=e["bertscore_model"], num_layers=12, lang="fa", batch_size=16)
        results.update(bertscore_p=round(float(P.mean()), 4), bertscore_r=round(float(R.mean()), 4),
                       bertscore_f1=round(float(F.mean()), 4))
    print(json.dumps(results, indent=2))
    (args.run / "eval.json").write_text(json.dumps(results, indent=2))
    with (args.run / "translations.tsv").open("w", encoding="utf-8") as fh:
        fh.write("source\treference\thypothesis\n")
        for s, r, h in zip(srcs, refs, hyps):
            fh.write(f"{s}\t{r}\t{h}\n")


if __name__ == "__main__":
    main()
