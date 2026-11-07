# Week 1: Pretraining (report material)

Draft material for the "Pretraining" section of the final report. All numbers
come from files in `week-1/results/` and `week-1/runs/`.

## Setup

- **Pretraining data:** Finnish Wikipedia, about 50,000 articles. After
  tokenization with my 32k BPE tokenizer: 47.5M training tokens and 0.44M
  validation tokens.
- **Fertility test data:** FLORES+ devtest split (parallel sentences), 1,012
  sentences each in English, Finnish and German.
- **Fertility** = number of tokens / number of whitespace-separated words.
  Whitespace words are not equally natural units in every language (Finnish
  packs more meaning into one word through case endings and compounds), so the
  numbers compare tokenizers within a language more fairly than across
  languages.
- **Hardware:** one NVIDIA GH200 GPU on CSC Roihu, bf16, seed 1234.

## A. Tokenizer fertility

Table 1. Tokens per word on FLORES+ (1,012 sentences per language). BPE
tokenizers trained on Finnish Wikipedia; Qwen and Gemma are the published
tokenizers.

| Tokenizer | English | Finnish | German |
|---|---:|---:|---:|
| BPE-8k (mine) | 2.59 | 2.34 | 3.25 |
| BPE-16k (mine) | 2.37 | 2.06 | 2.97 |
| BPE-32k (mine) | 2.19 | 1.84 | 2.75 |
| Qwen3.5-2B-Base | 1.26 | 2.83 | 1.66 |
| Gemma-4-E2B | 1.24 | 2.83 | 1.66 |

Observations:

- A bigger vocabulary gives fewer tokens per word in every language
  (Finnish 2.34 → 2.06 → 1.84 from 8k to 32k).
- My tokenizers are best on Finnish, the language they were trained on, and
  worst on German, which was not in the training data and has long compounds.
- The large multilingual tokenizers show the opposite pattern: English 1.26,
  but Finnish 2.83 tokens per word, about 2.2 times more. A Finnish text
  therefore costs more than twice as many tokens (compute, context length,
  API price) as an English text with the same number of words.
- Qwen and Gemma are almost identical, which suggests similar training
  data mixes dominated by English.

## B. Training one tiny GPT (medium configuration)

Table 2. Medium run settings and cost.

| Item | Value |
|---|---|
| Architecture | decoder-only GPT, 6 layers, 5 heads, width 320, context 512, tied embeddings |
| Parameters | 17.78M |
| Tokenizer | my BPE-32k |
| Tokens processed | 1.00B (15,259 steps × 65,536 tokens) |
| Passes over the training data | about 21 (1.0B / 47.5M) |
| Learning rate | 8e-4, warm-up 100 steps, decay to 8e-5 |
| Wall-clock time | 552 s on 1 GH200 (about 2M tokens/s) |
| Estimated compute | C ≈ 6ND = 6 × 17.78M × 1.0B ≈ 1.07 × 10^17 FLOPs |

Figure 1: `week-1/results/medium-loss.png` (training and validation loss).

Did the model learn?

- Loss starts at 10.42, which equals ln(32,000): the model guesses uniformly
  over the vocabulary. It ends at about 3.5 (training) and 3.93 (validation),
  so it predicts the next token far better than chance. A validation loss of
  3.93 corresponds to a perplexity of about e^3.93 ≈ 51.
- The validation loss falls steadily and then flattens around 3.90 for the
  last ~2,000 steps (values between 3.88 and 3.93). It never rises clearly,
  so there is no strong overfitting, even though the data was repeated
  about 21 times.
- There is a gap of about 0.4 between training and validation loss. The most
  likely reason is the repeated data: the model has seen each training
  sentence many times.
- The validation loss is estimated on 20 random batches, so single points
  are noisy (±0.03). The final value 3.93 is within that noise of the minimum
  3.88 at step 15,200.

What the experiment cannot show: low loss does not mean useful or factual
text (see Part D); one seed gives no error bars; validation loss only
measures Finnish Wikipedia-style text.

## C. Scaling (optional extension)

Table 3. Three model sizes, same tokenizer, data, seed and 1.0B-token budget.

| Run | Layers / width | Parameters | Compute (6ND) | Time | Final val. loss |
|---|---|---:|---:|---:|---:|
| small | 4 / 256 | 11.47M | 6.9 × 10^16 | 382 s | 4.137 |
| medium | 6 / 320 | 17.78M | 1.07 × 10^17 | 552 s | 3.934 |
| large | 8 / 384 | 26.65M | 1.60 × 10^17 | 737 s | 3.825 |

Figure 2: `week-1/results/scaling.png` (validation loss against compute).

- Larger models reach lower loss with the same data: −0.20 from small to
  medium, −0.11 from medium to large. Each step costs about 1.5 times more
  compute but gives a smaller improvement: diminishing returns, as scaling
  laws predict.
- Three points are an illustration, not a fitted scaling law, and with only
  47.5M unique tokens the larger models are partly limited by repeated data.
- By the Chinchilla rule of thumb (about 20 tokens per parameter), the
  compute-optimal budget for 17.8M parameters is about 0.36B tokens; I used
  1.0B (about 56 tokens per parameter), but most of these were repeats.

## D. What scale buys (optional extension)

Same few-shot English sentiment prompt (three labelled reviews, then
"Exactly what I wanted, and cheaper than the shop. Sentiment:"), greedy
decoding, 40 new tokens. File: `week-1/results/generation-comparison.json`.

| Model | Output (start) |
|---|---|
| my medium tiny GPT (17.8M, Finnish only) | "a warted by the warted by the warted by the …" |
| Qwen2.5-0.5B base (about 0.5B, multilingual) | "positive" — then continues with new invented reviews |

- The tiny model produces a repeating loop and does not follow the pattern.
  It never saw English text, so this is partly a language mismatch, not only
  a size difference. Architecture and data also differ, so the comparison is
  qualitative.
- The pretrained base model gets the label right (in-context learning) but
  keeps writing more examples: a base model continues text and does not know
  when to stop. This motivates instruction tuning (Week 2).

## Compute compared with a frontier model

My medium run used about 1 × 10^17 FLOPs. Meta reports about 3.8 × 10^25
FLOPs for training Llama 3.1 405B, roughly 4 × 10^8 times more. My run took 9
minutes on one GPU; at the same throughput the frontier run would need
thousands of years on a single GPU, which is why such runs use thousands of
GPUs for months.

## Files for the code archive

- Configs: `week-1/configs/{small,medium,large}.yaml`
- Results: `week-1/results/{fertility.csv, medium-loss.png, scaling.csv, scaling.png, generation-comparison.json}`
- Runs: `week-1/runs/{small,medium,large}/{metrics.jsonl, summary.json}`
