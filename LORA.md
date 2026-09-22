# LoRA notes — cheap adaptation on current hardware

Everything below runs on the present 2×A4000 box, plus hosted APIs and Hugging Face where weights come from. No new GPUs until a measured workload demands one: every technique here exists to fit training into memory already owned. Status: learning notes, not a runbook. Flowing prose, no hardwrap, same as the rest of this repo.

## LoRA: frozen base, learned side-path

Freeze the pretrained weights `W` and learn only an update `ΔW = B·A`, with `B` d×r and `A` r×k at rank r far below the dimensions (8–64 against thousands). Trainable parameters drop roughly 100–1000× versus full fine-tuning; the adapter file is megabytes, and merging it back into `W` costs zero latency at inference. What LoRA cannot do is the actual decision rule: rank is a bottleneck, so it adapts well to tasks near pretraining (style, format, domain flavor, tool use) and poorly to capabilities far outside it (new reasoning, new languages, new modalities). Full fine-tuning stays the answer for small models — a 600M model trains end-to-end in ~10 GB, so LoRA would save almost nothing there.

## QLoRA: the same, under quantization

Quantize the frozen base to 4-bit (NF4) and train LoRA adapters on top in bf16. An 8B instruct model fits in roughly 12–20 GB: the standard shape for adapting big models on one card. Double quantization and paged optimizers (both in bitsandbytes) shave further, at small speed cost. Watch the failure modes: quantization error compounds on already-quantized or tiny models, and 4-bit base weights plus bf16 adapters need the compute dtype set deliberately or training silently degrades.

## DoRA and the family (which D matters)

DoRA (Weight-Decomposed Low-Rank Adaptation, https://arxiv.org/abs/2402.10969) splits each weight into magnitude and direction, learns the magnitude fully and the direction by LoRA. It closes much of the gap to full fine-tuning at small extra cost — the first variant to try when plain LoRA underperforms. If you meant a different D: DyLoRA trains one adapter usable across ranks (train once, slice rank at serve time); distributed LoRA is FSDP plus adapters for multi-GPU full-ish runs; diffusion LoRAs (image models) are a separate community with its own tooling and are not covered here. The rest of the family, one line each: AdaLoRA budgets rank per layer adaptively; rsLoRA stabilizes higher ranks with a corrected scaling factor; LoRA+ uses a larger learning rate on `B` than `A`; VeRA freezes random projections and learns only scaling vectors (tiny adapters, weaker); PiSSA initializes adapters from the principal components instead of random noise (faster convergence); LoHA/LoKr replace the product with Hadamard/Kronecker structure (more expressivity per parameter, fussier).

## Memory math that decides the method

Full fine-tuning with Adam holds roughly 16 bytes per parameter (weights, gradients, two optimizer states, mixed precision overhead more) — an 8B needs on the order of 128 GB, which is why it never happens on one card. LoRA trains under 1% of parameters, so optimizer states shrink to the adapters; activations dominate instead, and gradient checkpointing trades roughly 30% more compute for the bulk of activation memory back. QLoRA adds the 4-bit base on top. Worked examples: 0.6B full fine-tune ≈ 10 GB (fits anywhere, skip LoRA); 8B QLoRA rank-64 ≈ 15–20 GB (one A4000, tight but routine); 8B bf16 LoRA ≈ 25–35 GB (needs both cards or a bigger one); 7B VLM QLoRA ≈ 16–24 GB (same, plus image tokens in context).

## Knobs that actually move results

Rank 16–64 to start (higher for structured output and code, lower for style); alpha at roughly twice rank (the community default that keeps effective step size sane); target all attention projections plus the MLP (`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` on Llama-likes — adapters on attention alone underperform measurably); learning rate near 2e-4 for QLoRA with a short warmup; gradient checkpointing on unless VRAM is idle; batch size 4–16 with accumulation to taste (small batches are noisier but generalize fine at these scales). None of these substitute for data: a clean thousand pairs beats a noisy hundred thousand, and the eval split from the decider work (held-out cases, Brier and ECE alongside accuracy) applies unchanged.

## Calibration is not optional for gates

Both Jev-class projects shipped over-confident and fixed it the same way: one temperature per question type (and option count), fit on held-out calibration cases. Any adapter whose probabilities gate actions — tool calls, merges, human escalation — gets the same treatment, or the gate is theater. Probabilities you never act on can stay uncalibrated; probabilities you act on cannot.

## Learning resources

Papers first, then tooling: LoRA (https://arxiv.org/abs/2106.09685), QLoRA (https://arxiv.org/abs/2305.14314), DoRA (https://arxiv.org/abs/2402.10969). Libraries: Hugging Face PEFT (https://github.com/huggingface/peft) for every variant above, bitsandbytes (https://github.com/bitsandbytes-foundation/bitsandbytes) for the quantization underneath, Unsloth (https://github.com/unslothai/unsloth, docs at https://docs.unsloth.ai) for the fastest practical QLoRA loop, Axolotl (https://github.com/axolotl-ai-cloud/axolotl) when configs outgrow notebooks, Liger kernels (https://github.com/linkedin/Liger-Kernel) for memory-efficient fused ops. Memory mechanics: PyTorch activation checkpointing (https://pytorch.org/docs/stable/checkpoint.html), FSDP when one card stops being enough (https://pytorch.org/docs/stable/fsdp.html), DeepSpeed ZeRO for the rest (https://github.com/microsoft/DeepSpeed). Directly applicable prior art: the Laya fine-tune notebook that runs the full loop on free 2×T4 hardware (https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb), and the open Jev reproductions catalogued at https://github.com/mihado/awesome-jev.
