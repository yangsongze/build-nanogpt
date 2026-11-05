# Learning guide

This guide connects the course implementation to the small local demo. The explanations describe the code; they are not a record of experiments beyond the checked-in run.

## From text to predictions

The demo maps each distinct character in `input.txt` to an integer. The course’s full training setup instead uses GPT-2’s byte-pair encoding tokenizer. Both produce token IDs, but their vocabulary sizes and token meanings differ.

For each training window, the target is the same sequence shifted one position forward. Given `ROMEO`, for example, input characters `ROME` have targets `OMEO`. The network outputs a distribution over the vocabulary at every position; cross-entropy penalizes assigning low probability to the correct next token.

## Inside the Transformer

Token embeddings provide learned vectors for token IDs. Position embeddings add information about where each token appears in the context window. Each block applies layer normalization, causal attention, a residual connection, then another normalized feed-forward transformation and residual connection.

Attention forms queries, keys, and values. Multiple heads let the model combine different attention patterns. The causal mask prevents future-token information from entering a prediction. Without that constraint, the shifted-target training task could leak the answer.

The final layer normalization and vocabulary projection produce logits. The input embedding and output projection share weights. The model uses residual projection initialization scaled with depth to control the size of updates through the residual stream.

## Training and evaluation

The demo uses AdamW and gradient clipping. It splits the text sequentially into 90% training data and 10% validation data. Validation runs without gradients and uses fixed sampled windows, making comparisons across training steps easier to interpret. These windows estimate held-out loss rather than evaluating every validation token.

The full course script adds gradient accumulation, learning-rate warmup and cosine decay, distributed training, larger data shards, mixed precision, and HellaSwag evaluation. Those techniques are present in the course code; the local demo does not exercise them.

A lower training loss shows improved fit to training windows. A lower validation loss provides evidence of improvement on held-out text. Readable samples are useful qualitative evidence, but a single attractive sample cannot establish general model quality.

## Generation

The demo starts with `ROMEO:` and repeatedly samples one next character from the model’s predicted distribution. Each prediction uses at most the last 64 characters. Sampling is stochastic, and short training runs typically produce noisy text. This is a text-completion model; it has no instruction tuning or chat behavior.

## Experiments to try

1. Compare 50, 200, and 1,000 training steps with the same seed. Record training and validation loss as well as sample text.
2. Change `block_size` and consider which dependencies can fit within the new context window.
3. Change `n_layer` or `n_embd` and compare parameter counts, runtime, and validation loss.
4. Compare multiple seeds before drawing conclusions about a small loss difference.

Save each run to a different `--output` path. When sharing results, include the command, hardware, software version, configuration, and limitations. Only describe experiments as completed after running them.
