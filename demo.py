"""Small character-level training demo using the course's GPT architecture."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from train_gpt2 import GPT, GPTConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--generate', type=int, default=300, help='characters to sample')
    parser.add_argument('--output', type=Path, default=Path('outputs/demo.json'))
    args = parser.parse_args()
    if args.steps < 1 or args.generate < 1:
        parser.error('--steps and --generate must be positive')

    torch.manual_seed(args.seed)
    # Small matrix operations can be slower with many CPU worker threads.
    if args.device == 'cpu':
        torch.set_num_threads(1)
    text = Path(__file__).with_name('input.txt').read_text(encoding='utf-8')
    chars = sorted(set(text))
    stoi = {char: i for i, char in enumerate(chars)}
    data = torch.tensor([stoi[char] for char in text], dtype=torch.long)
    split = int(0.9 * len(data))
    train, val = data[:split], data[split:]
    config = GPTConfig(block_size=64, vocab_size=len(chars), n_layer=2, n_head=2, n_embd=64)
    model = GPT(config).to(args.device)
    optimizer = model.configure_optimizers(0.1, 1e-3, args.device)
    # Separate evaluation RNG keeps evaluation from changing training batches.
    train_rng = torch.Generator().manual_seed(args.seed)

    def batch(tokens, rng):
        starts = torch.randint(len(tokens) - config.block_size, (8,), generator=rng)
        x = torch.stack([tokens[i:i + config.block_size] for i in starts])
        y = torch.stack([tokens[i + 1:i + config.block_size + 1] for i in starts])
        return x.to(args.device), y.to(args.device)

    @torch.no_grad()
    def evaluate():
        model.eval()
        result = {}
        for name, tokens in [('train', train), ('val', val)]:
            rng = torch.Generator().manual_seed(args.seed + 1)
            result[name] = sum(model(*batch(tokens, rng))[1].item() for _ in range(10)) / 10
        model.train()
        return result

    initial = evaluate()
    history = [{'step': 0, **initial}]
    print(f"step 0 | train {initial['train']:.4f} | val {initial['val']:.4f}")
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(*batch(train, train_rng))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 50 == 0 or step == args.steps:
            metrics = evaluate()
            history.append({'step': step, **metrics})
            print(f"step {step} | train {metrics['train']:.4f} | val {metrics['val']:.4f}")

    model.eval()
    prompt = 'ROMEO:'
    tokens = torch.tensor([[stoi[c] for c in prompt]], device=args.device)
    with torch.no_grad():
        for _ in range(args.generate):
            logits, _ = model(tokens[:, -config.block_size:])
            next_token = torch.multinomial(F.softmax(logits[:, -1, :], dim=-1), 1)
            tokens = torch.cat((tokens, next_token), dim=1)
    sample = ''.join(chars[i] for i in tokens[0].tolist())
    result = {
        'description': 'Tiny character-level course demo; not a GPT-2 reproduction benchmark.',
        'torch_version': torch.__version__, 'device': args.device, 'seed': args.seed,
        'steps': args.steps, 'batch_size': 8, 'learning_rate': 1e-3,
        'config': asdict(config), 'parameters': sum(p.numel() for p in model.parameters()),
        'metrics': history, 'prompt': prompt, 'sample': sample,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f'\n{sample}\n\nRun record saved to {args.output}')


if __name__ == '__main__':
    main()
