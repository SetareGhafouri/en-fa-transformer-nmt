"""Corpus loading, normalisation, vocabulary construction and tensorisation."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PAD, UNK, SOS, EOS = "<pad>", "<unk>", "<sos>", "<eos>"
SPECIALS = [PAD, UNK, SOS, EOS]
PARSINLU = "hf://datasets/persiannlp/parsinlu_translation_en_fa@refs/convert/parquet/parsinlu-repo"

_DIACRITICS = re.compile("[\u064B-\u065F\u0670\u06D6-\u06ED]")  # Arabic diacritics


def _strip_control(text: str) -> str:
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch == "‌")


def clean_en(text: str | None) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    return re.sub(r"\s+", " ", _strip_control(text)).strip()


def clean_fa(text: str | None) -> str:
    """Unify Arabic/Persian letter variants, drop diacritics, keep the zero-width non-joiner."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u064A", "\u06CC").replace("\u0643", "\u06A9")  # Arabic yeh/kaf -> Persian
    text = _DIACRITICS.sub("", text)
    return re.sub(r"[ \t\r\f\v]+", " ", _strip_control(text)).strip()


def load_pairs(fraction: float, seed: int) -> dict[str, list[tuple[str, str]]]:
    """Sample ``fraction`` of the ParsiNLU training corpus and split it 80/10/10."""
    from datasets import load_dataset

    raw = load_dataset("parquet", data_files=f"{PARSINLU}/train/*.parquet", split="train")
    raw = raw.shuffle(seed=seed).select(range(int(len(raw) * fraction)))
    pairs = []
    for ex in raw:
        src, tgts = clean_en(ex["source"]), ex["targets"]
        tgt = clean_fa(tgts[0]) if tgts else ""
        if src and tgt:
            pairs.append((src, tgt))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    n_tr, n_va = int(0.8 * len(idx)), int(0.9 * len(idx))
    take = lambda ids: [pairs[i] for i in ids]
    return {"train": take(idx[:n_tr]), "val": take(idx[n_tr:n_va]), "test": take(idx[n_va:])}


@dataclass
class Vocab:
    itos: list[str]

    def __post_init__(self):
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, sentences: list[str], size: int, min_freq: int) -> "Vocab":
        counts = Counter(tok for s in sentences for tok in s.split())
        words = [w for w, c in counts.most_common() if c >= min_freq][: size - len(SPECIALS)]
        return cls(SPECIALS + words)

    def encode(self, tokens: list[str]) -> list[int]:
        unk = self.stoi[UNK]
        return [self.stoi.get(t, unk) for t in tokens]

    def decode(self, ids) -> str:
        out = []
        for i in ids:
            w = self.itos[int(i)]
            if w == EOS:
                break
            if w not in (PAD, SOS):
                out.append(w)
        return " ".join(out)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.itos, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        return cls(json.loads(path.read_text()))


def pad(ids: list[int], length: int, pad_id: int = 0) -> list[int]:
    ids = ids[:length]
    return ids + [pad_id] * (length - len(ids))


def tensorise(pairs, src_vocab: Vocab, tgt_vocab: Vocab, src_len: int, tgt_len: int):
    """Encoder input, decoder input (<sos> + y) and shifted target (y + <eos>) for teacher forcing."""
    sos, eos = tgt_vocab.stoi[SOS], tgt_vocab.stoi[EOS]
    enc, dec_in, dec_out = [], [], []
    for src, tgt in pairs:
        y = tgt_vocab.encode(tgt.split())
        enc.append(pad(src_vocab.encode(src.split()), src_len))
        dec_in.append(pad([sos] + y, tgt_len))
        dec_out.append(pad(y + [eos], tgt_len))
    as_arr = lambda a: np.asarray(a, dtype=np.int32)
    return as_arr(enc), as_arr(dec_in), as_arr(dec_out)


def length_cutoff(sentences: list[str], percentile: float) -> int:
    return max(5, int(np.percentile([len(s.split()) for s in sentences], percentile)))


def fasttext_matrix(vec_path: str, vocab: Vocab, dim: int = 300, seed: int = 42) -> tuple[np.ndarray, float]:
    """Embedding matrix initialised from FastText ``.vec`` vectors; returns (matrix, coverage)."""
    rng = np.random.default_rng(seed)
    E = rng.normal(0, 0.01, size=(len(vocab), dim)).astype(np.float32)
    E[: len(SPECIALS)] = 0.0
    found = 0
    with open(vec_path, encoding="utf-8", errors="ignore") as fh:
        next(fh)  # header: "<count> <dim>"
        for line in fh:
            word, _, rest = line.rstrip().partition(" ")
            idx = vocab.stoi.get(word)
            if idx is not None and idx >= len(SPECIALS):
                vec = np.fromstring(rest, sep=" ", dtype=np.float32)
                if vec.size == dim:
                    E[idx] = vec
                    found += 1
    return E, found / len(vocab)
