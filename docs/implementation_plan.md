# DUET-Grow implementation plan

## Phase 0 — paper fixes

Use the corrected design, not the first report draft:

- width features are projected off rowspace(A^l);
- width SVD proposes incoming weights but the real nonlinear features are evaluated and outgoing weights are refit;
- depth is an MLP side branch `z^{l+1} = W^{l+1}a^l+b+γ W_out σ(W_b a^l)` or a residual-stream block for ResNets;
- the depth estimate is the realized gain of the selected h units, not the `H`-candidate SVD gain;
- use a significance-based or fixed-overhead size rule instead of `argmax G(K)/(cK)`;
- use `ω = ω_learn + γ ω_target` so gradients exist at birth.

## Phase 1 — falsification

Before CIFAR/ImageNet, run the depth-vs-width synthetic experiment. The research plan explicitly puts this first.

## Phase 2 — static score validation

Train a small MLP. Capture one large batch of `A^{l-1}`, `A^l`, and `Δ`. Check:

`R = Δ - δW* A^l`

then verify that the analytic width proposal and nonlinear refit produce a measurable held-out gain.

## Phase 3 — dynamic MLP

Use `GrowableMLP` in this repository. At a growth event:

1. train to plateau;
2. collect a score batch and a separate held-out batch;
3. compute width/depth actions for each hidden layer;
4. reject layers failing the permutation signal test;
5. choose the highest realized-gain/FLOP action;
6. grow physically;
7. ramp gamma from 0 to 1;
8. rebuild optimizer state carefully;
9. continue training.

## Phase 4 — CIFAR

Do CIFAR-10, then CIFAR-100. First use a thin MLP prototype, then replace the fixed feature extractor with a small CNN.

## Phase 5 — ImageNet

Use `ImageFolder` and start with a fixed, reproducible subset. Only after the CNN growth code is validated should you run full ImageNet-1K.

## Phase 6 — ResNet18

For the final vision result, implement:

- im2col statistics over convolution patches;
- channel growth with adjacent convolution/BN surgery;
- residual-stream depth insertion `a <- a + γ U σ(W_b a)`;
- projection/shortcut handling;
- tensor padding accounting;
- optimizer-state surgery;
- equal-FLOP comparisons.

## Metrics

Track accuracy, loss, added FLOPs, wall-clock time, parameter count, energy, top Hessian eigenvalue/SAM sharpness, and dead-neuron/channel fraction.
