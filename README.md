# UMAY

A bilingual (Turkish–English) language model built from scratch.

*Umay is a protective spirit in Turkic mythology.*

## UMAY-0

A character-level GPT — a hand-written transformer, built for learning.

- 6 layers, 6 heads, 384 embedding dim, 256 context
- ~10.8M parameters
- Data: mixed Turkish + English Wikipedia (60/40)
- First run: val loss 1.31 after 5000 steps (~3 min on an RTX 5070 Ti)

## Setup

```
uv venv --python 3.12
.venv\Scripts\activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install numpy matplotlib datasets tqdm
```

## Usage

```
python data_download.py   # download the corpus
python tokenizer.py       # build vocab, produce char.pt
python train.py           # train
python sample.py          # generate text
```

## Roadmap

- [x] UMAY-0 — character-level, from scratch
- [ ] Custom BPE tokenizer trained on the mixed corpus
  -[x] Merge algorithm from scratch (2.43x compression at 1k merges)
- [ ] Longer training, LR schedule, `torch.compile`
- [ ] UMAY-1 — a conversational assistant