# TransForge — Seq2Seq Translation Engine Built Atom by Atom
### A ground-up reconstruction of "Attention Is All You Need" · PyTorch · DE→EN
 
> Most people use transformers. This one was *built*.
> No `nn.Transformer`. No tutorial clones. Every tensor operation — from scaled dot-product attention to sinusoidal positional encoding — derived, written, and trained from first principles.
 
---
 
## What This Is
 
TransForge is a research-grade reimplementation of the original Transformer architecture (Attention Is All You Need), applied to German→English neural machine translation on the Multi30k dataset. The objective was not to maximize BLEU — it was to achieve complete mechanical understanding of every component before touching a framework abstraction. Every class, every matrix multiplication, every masking decision is intentional and explainable.
 
---
 
## Architecture
 
| Component | Detail |
|---|---|
| Model | Encoder-Decoder Transformer |
| `d_model` | 256 |
| `n_heads` | 8 |
| `n_layers` | 3 (each encoder + decoder) |
| FFN hidden size | 512 |
| Max sequence length | 128 |
| Dropout | 0.1 |
| Trainable parameters | ~10M (varies with vocab size) |
 
**Built from scratch:**
- `ScaledDotProductAttention` — score = QKᵀ / √d, softmax, weighted sum of V
- `MultiHeadAttention` — 8 parallel attention heads, split → attend → concat → project
- `PositionalEncoding` — sinusoidal, fixed (sin for even dims, cos for odd dims)
- `TokenEmbedding` — learned embeddings with pad masking
- `LayerNorm` — manual implementation with learnable γ, β
- `PositionwiseFeedForward` — Linear → ReLU → Dropout → Linear
- `EncoderLayer` — self-attention + residual + norm + FFN + residual + norm
- `DecoderLayer` — masked self-attention + cross-attention + FFN, each with residuals
- `Encoder` / `Decoder` — stacked N layers
- `Transformer` — source + target masking, full forward pass
---
 
## Dataset
 
**Multi30k** (`bentrevett/multi30k` via HuggingFace Datasets)
 
- ~29,000 train / ~1,000 val / ~1,000 test sentence pairs
- Tokenized with spaCy (`de_core_news_sm`, `en_core_web_sm`)
- Vocabulary built with `min_freq=2`, specials: `<unk>`, `<pad>`, `<sos>`, `<eos>`
---
 
## Training Setup
 
| Hyperparameter | Value |
|---|---|
| Optimizer | Adam (β₁=0.9, β₂=0.98, ε=1e-9) |
| Learning rate | 1e-4 |
| Loss | CrossEntropyLoss (ignores `<pad>`) |
| Gradient clipping | 1.0 |
| Epochs | 20 |
| Batch size | 128 |
 
Model checkpoint saved at best validation loss.
 
---
 
## Sample Translations
 
```
DE: ein mann spielt gitarre .
EN: a man plays guitar .
 
DE: eine frau läuft durch den park .
EN: a woman walks through the park .
 
DE: zwei kinder spielen im schnee .
EN: two children play in the snow .
 
DE: ein hund rennt über das feld .
EN: a dog runs across the field .
```
 
---
 
## How to Run
 
```bash
# Install dependencies
pip install torch datasets spacy torchtext
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
 
# Run the notebook
jupyter notebook de-to-en.ipynb
```
 
---
