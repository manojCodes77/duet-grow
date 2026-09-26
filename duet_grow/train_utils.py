from __future__ import annotations

import torch


def train_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    total = correct = seen = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item()) * y.size(0)
        correct += int((logits.argmax(1) == y).sum())
        seen += y.size(0)
    return loss_sum / seen, correct / seen


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    total = correct = seen = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        total += float(loss_fn(logits, y).item()) * y.size(0)
        correct += int((logits.argmax(1) == y).sum())
        seen += y.size(0)
    return total / seen, correct / seen
