import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from duet_grow.data import imagenet_loaders
from duet_grow.model import GrowableMLP
from duet_grow.train_utils import evaluate, train_epoch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    train, val, ncls = imagenet_loaders(args.data, batch_size=args.batch_size)
    model = GrowableMLP(224 * 224 * 3, [256, 256], ncls).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

    class Flat(torch.utils.data.Dataset):
        def __init__(self, base): self.base = base
        def __len__(self): return len(self.base)
        def __getitem__(self, i):
            x, y = self.base[i]
            return x.flatten(), y
    train_f = torch.utils.data.DataLoader(Flat(train.dataset), batch_size=args.batch_size, shuffle=True, num_workers=train.num_workers)
    val_f = torch.utils.data.DataLoader(Flat(val.dataset), batch_size=args.batch_size, shuffle=False, num_workers=val.num_workers)
    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_epoch(model, train_f, opt, args.device)
        va_loss, va_acc = evaluate(model, val_f, args.device)
        print(f"epoch={epoch+1} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.4f}")


if __name__ == "__main__":
    main()
