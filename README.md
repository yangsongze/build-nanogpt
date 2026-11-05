# Building nanoGPT — my course project

A learning portfolio project from my walkthrough of Andrej Karpathy’s **[Let's reproduce GPT-2 (124M)](https://youtu.be/l8pRSuU81PU)** lecture. I completed the walkthrough and use this repository to demonstrate the concepts covered: building a decoder-only Transformer, training it to predict the next token, and evaluating its outputs.

The core implementation comes from **[karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt)**. This is a course-based learning project. The original commit history is retained; those upstream commits belong to their original authors.

## Try the local demo

Train a small model on the included Shakespeare text. This uses the course’s GPT architecture with a character vocabulary and a much smaller configuration, so it can run on a CPU without downloading a dataset or pretrained weights.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python demo.py
```

Python 3.10 or later is required. Dependency installation requires internet access; the demo itself runs offline. On Windows, activate with `.venv\Scripts\activate`.

For a quick check or a longer experiment:

```bash
python demo.py --steps 10 --generate 100
python demo.py --steps 1000 --device mps
python demo.py --steps 1000 --device cuda
```

The default device is CPU. Choose `mps` for a supported Apple Silicon setup or `cuda` for an available NVIDIA GPU. Runtime depends on hardware.

The demo prints training and held-out validation loss, then samples text from the prompt `ROMEO:`. It saves configuration, metrics, seed, and generated text to `outputs/demo.json`. Early samples will be rough: the goal is to show a working training pipeline and measurable learning.

## Recorded experiment

A checked-in [run record](docs/demo-run.json) contains the actual output of:

```bash
python demo.py --steps 200 --output docs/demo-run.json
```

The recorded CPU run uses **108,352 parameters**. Over 200 steps, training loss fell from **4.2007 to 2.5805** and validation loss from **4.1922 to 2.5486** (cross-entropy in nats per character). Generated text remains rough.

This is a small character-level experiment, separate from the course’s 124M-parameter GPT-2 setup. Its loss values and generated text should not be compared directly with GPT-2 or HellaSwag results. The record includes the PyTorch version and device; exact results may vary across hardware and library versions.

## What this project covers

| Concept | Where to look | What to explain |
| --- | --- | --- |
| Token and position embeddings | `GPT` in `train_gpt2.py` | How token identity and sequence position enter the model |
| Causal multi-head attention | `CausalSelfAttention` | Why a token cannot attend to future tokens |
| Transformer blocks | `Block`, `MLP` | How residual connections, layer normalization, and feed-forward layers fit together |
| Next-token prediction | `GPT.forward`, `demo.py` | How shifted targets and cross-entropy define the training task |
| Optimization | `configure_optimizers`, course training loop | AdamW, weight decay, clipping, accumulation, and learning-rate scheduling |
| Evaluation and sampling | `demo.py`, `hellaswag.py` | Why validation loss, multiple-choice accuracy, and sample quality measure different things |
| Scaling training | course training loop | How data shards and distributed data parallelism support larger runs |

See the [learning guide](docs/LEARNING_GUIDE.md) for a short explanation and suggested experiments.

## My additions

- This portfolio README and a guide to the concepts covered in the course.
- `demo.py`: a configurable laptop demo with training, validation, text sampling, and a saved run record.
- An import guard and deferred course-only imports in `train_gpt2.py`, allowing the demo to reuse `GPT` without launching the full training job.
- Dependency files and ignore rules for generated data, logs, and checkpoints.

## Repository map

| File | Purpose |
| --- | --- |
| `demo.py` | Small local training and generation demo |
| `train_gpt2.py` | Upstream GPT architecture and full course training loop |
| `input.txt` | Included Shakespeare text used by the local demo |
| `fineweb.py` | Course FineWeb-Edu download and tokenization script |
| `hellaswag.py` | Course multiple-choice evaluation utility |
| `play.ipynb` | Upstream experiment notebook |
| `docs/UPSTREAM_README.md` | Original README, including upstream examples and errata |

## Full course workflow

Install the additional dependencies with `python -m pip install -r requirements-course.txt`.

The original workflow uses `python fineweb.py` to prepare the 10B-token FineWeb-Edu sample, followed by `python train_gpt2.py` (or `torchrun --standalone --nproc_per_node=8 train_gpt2.py` for eight CUDA GPUs). Read the script settings before running: this downloads a large dataset and launches the full training configuration. It is separate from the lightweight demo above.

The original README’s sample generations and benchmark discussion describe upstream work. This repository does not claim that I independently trained GPT-2 at that scale or achieved those results. Completing a public lecture walkthrough is self-directed study, not a formal course certificate.

## Acknowledgments

Credit to Andrej Karpathy and the upstream contributors for the implementation and teaching material. The upstream README states MIT as the code license; its original text is preserved in [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md). Dataset and dependency terms belong to their respective sources.
