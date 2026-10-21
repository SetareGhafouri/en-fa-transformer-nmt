# English→Persian Neural Machine Translation with a Transformer

A from-scratch Transformer encoder–decoder for **English-to-Persian** translation, trained on the
ParsiNLU parallel corpus. Token embeddings are initialised from pre-trained FastText vectors.
The project also takes a close look at a less discussed question: **how to evaluate machine
translation into a non-Latin-script, morphologically rich language** without fooling yourself.

> Developed from a graduate Deep Learning course assignment at Isfahan University of Technology,
> then rewritten as a modular, configuration-driven codebase with a corrected evaluation protocol.

## Setup

| | |
|---|---|
| **Data** | [ParsiNLU](https://huggingface.co/datasets/persiannlp/parsinlu_translation_en_fa) EN→FA (1.62 M pairs). A random 20% is used and split 80/10/10: 259 K train / 32 K validation / 32 K test |
| **Normalisation** | NFKC; Arabic ي/ك mapped to Persian ی/ک; diacritics removed; zero-width non-joiner (نیم‌فاصله) preserved |
| **Vocabulary** | word level, 30 K per language (min. frequency 2); sequences truncated at the 95th length percentile (23 EN / 25 FA tokens) |
| **Embeddings** | FastText `cc.*.300`, frozen; vocabulary coverage 72.4% (EN) and 75.9% (FA) |
| **Model** | 2 encoder + 2 decoder layers, d = 300, 6 heads, FFN 1200, dropout 0.1, learned positions; padding and causal masks |
| **Training** | masked cross-entropy, Adam (1e-4), batch 64, early stopping on validation loss; ≈ 3 min/epoch on a P100 |
| **Decoding** | greedy |

## Results

On 2,000 held-out test sentences, after 10 epochs (original experiment):

| Metric | Score |
|---|---:|
| SacreBLEU | 9.5 (1–4-gram precision 34.1 / 14.1 / 7.0 / 2.4) |
| BERTScore F1 (ParsBERT, 1,000 sentences) | 0.582 |
| Validation token accuracy | 33.3% |

Validation loss was still decreasing at epoch 10 (4.03 → 4.01), so the model had **not converged**.

### Example translations

| Source | Reference | Model output |
|---|---|---|
| he glanced at me, and mr. … hand gently patted me upon the shoulder. | آقای مل نگاهی به من کرد و دستش را … بر شانه‌ام نواخت. | به من نگاه کرد و آقای لاری دستش را روی شانه‌ی |
| that if she had accidentally fallen in the water | اینکه اگر ربرتا تصادفاً به درون آب افتاده بود، | اگر او در آب فرو رفته بود |
| when i'm eating, that's all i think about. if i'm on the march, i just concentrate on … | وقتی در حال … فقط به خوردن فکر می‌کنم … | وقتی من فکر می‌کنم که اگر … تمرکز کنم و تمرکز کنم … |

The model captures sentence structure and common phrases well. The two characteristic failures are
**repetition loops** in long sentences (a known pathology of greedy decoding) and
**out-of-vocabulary words**, especially proper names.

## Evaluating translation into Persian: three pitfalls

Re-examining the evaluation turned out to be the most instructive part of the project.

**1. ROUGE silently ignores Persian text.** The `rouge_score` package's default tokenizer
lower-cases and then deletes every character outside `[a-z0-9]`. A Persian sentence such as
`وقتی من فکر میکنم <unk> که` is reduced to the single token `unk`. ROUGE computed this way
measures nothing except the overlap of unknown-word markers. That is why the original run
reported ROUGE-1 = 0.193 and **ROUGE-L exactly equal to ROUGE-1**. The scores were an artefact,
so they are not reported above. `src/evaluate.py` uses a whitespace tokenizer instead.

**2. References must be the real sentences.** The original pipeline rebuilt the reference
translations from the model's own target ids. Those references had already been truncated to
25 tokens, and every out-of-vocabulary word had been replaced by `<unk>`. The model is then
rewarded for producing `<unk>` exactly where the reference has one. The corrected evaluation
scores against the original cleaned Persian sentences.

**3. Prefer character-level and embedding-based metrics.** Persian is morphologically rich and
has inconsistent spacing (for example `میکنم` vs. `می‌کنم`). Word-level BLEU therefore
penalises translations that are correct but spelled differently. The evaluation script also
reports **chrF**, which is more robust to this. BERTScore with ParsBERT captures meaning, but it
is reported here without baseline rescaling, so its absolute value is only meaningful when
comparing systems.

Re-running `src/evaluate.py` gives the methodologically clean numbers.

## Limitations and next steps

* **Subword vocabulary.** With a 30 K word vocabulary, about a quarter of the word types have no
  FastText vector, and rare words become `<unk>`. A shared SentencePiece/BPE vocabulary would
  remove the unknown-word problem almost entirely and is the single largest expected gain.
* **Decoding.** Beam search with length normalisation would reduce the repetition loops.
* **Scale and training.** The model is small (2+2 layers) and was stopped before convergence.
  A learning-rate warm-up schedule and longer training are standard improvements.
* **Frozen embeddings.** Unfreezing the FastText vectors after a few epochs is a cheap ablation.
* **Comparison.** Fine-tuning a pre-trained multilingual model (e.g. mT5, NLLB) would show how far
  a from-scratch model is from the current state of the art.

## Usage

```bash
pip install -r requirements.txt
# FastText vectors (about 1.2 GB each), placed in vectors/:
#   https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz
#   https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fa.300.vec.gz
python src/train.py    --config configs/base.yaml --out checkpoints/base
python src/evaluate.py --config configs/base.yaml --run checkpoints/base
```

The ParsiNLU corpus is downloaded automatically from the Hugging Face Hub. `evaluate.py` writes
`eval.json` and a `translations.tsv` file with every source, reference and hypothesis.

## Structure

```
configs/base.yaml     all hyperparameters
src/nmt/data.py       normalisation, vocabulary, tensorisation, FastText loading
src/nmt/model.py      Transformer encoder/decoder blocks, masked loss, batched greedy decoding
src/train.py          training with checkpointing and early stopping
src/evaluate.py       BLEU, chrF, ROUGE (Persian-safe), BERTScore
```

## References

* A. Vaswani et al. *Attention Is All You Need.* NeurIPS 2017.
* P. Bojanowski et al. *Enriching Word Vectors with Subword Information.* TACL 2017.
* D. Khashabi et al. *ParsiNLU: A Suite of Language Understanding Challenges for Persian.* TACL 2021.
* M. Post. *A Call for Clarity in Reporting BLEU Scores.* WMT 2018.
* T. Zhang et al. *BERTScore: Evaluating Text Generation with BERT.* ICLR 2020.

## License

MIT
