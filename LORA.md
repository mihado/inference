# LoRA notes — cheap adaptation on current hardware

- Everything below runs on the present 2×A4000 box, plus hosted APIs and Hugging Face where weights come from.
- No new GPUs until a measured workload demands one: every technique here exists to fit training into memory already owned.
- Status: learning notes, not a runbook.

## LoRA: frozen base, learned side-path

- Freeze the pretrained weights `W`; learn only `ΔW = B·A` with rank r far below the dimensions (8–64 against thousands).
- Trainable parameters drop roughly 100–1000× versus full fine-tuning; the adapter file is megabytes.
- Merging the adapter back into `W` costs zero latency at inference.
- Rank is a bottleneck: LoRA adapts well near pretraining (style, format, domain flavor, tool use) and poorly far outside it (new reasoning, languages, modalities).
- Full fine-tuning stays the answer for small models: a 600M model trains end-to-end in ~10 GB, so LoRA would save almost nothing there.

## QLoRA: the same, under quantization

- Quantize the frozen base to 4-bit (NF4); train LoRA adapters on top in bf16.
- An 8B instruct model fits in roughly 12–20 GB: the standard shape for adapting big models on one card.
- Double quantization and paged optimizers shave further at small speed cost.
- Failure modes: quantization error compounds on already-quantized or tiny models; set the compute dtype deliberately or training silently degrades.

## DoRA and the family (which D matters)

- DoRA (Weight-Decomposed Low-Rank Adaptation, https://arxiv.org/abs/2402.10969) splits each weight into magnitude and direction, learns magnitude fully and direction by LoRA — closes much of the gap to full fine-tuning, and the first variant to try when plain LoRA underperforms.
- Other D candidates: DyLoRA trains one adapter usable across ranks (train once, slice at serve time); distributed LoRA is FSDP plus adapters for multi-GPU runs; diffusion LoRAs are a separate community, not covered here.
- The rest, one line each:
  - AdaLoRA budgets rank per layer adaptively.
  - rsLoRA stabilizes higher ranks with a corrected scaling factor.
  - LoRA+ uses a larger learning rate on `B` than `A`.
  - VeRA freezes random projections and learns only scaling vectors (tiny adapters, weaker).
  - PiSSA initializes from principal components instead of noise (faster convergence).
  - LoHA/LoKr use Hadamard/Kronecker structure (more expressivity per parameter, fussier).

## Memory math that decides the method

- Full fine-tuning with Adam holds roughly 16 bytes per parameter: an 8B needs on the order of 128 GB and never happens on one card.
- LoRA trains under 1% of parameters, so optimizer states shrink to the adapters; activations dominate instead.
- Gradient checkpointing trades roughly 30% more compute for the bulk of activation memory back.
- Worked examples: 0.6B full fine-tune ≈ 10 GB (fits anywhere, skip LoRA); 8B QLoRA rank-64 ≈ 15–20 GB (one A4000, tight but routine); 8B bf16 LoRA ≈ 25–35 GB (both cards or a bigger one); 7B VLM QLoRA ≈ 16–24 GB (plus image tokens in context).

## Knobs that actually move results

- Rank 16–64 to start (higher for structured output and code, lower for style).
- Alpha at roughly twice rank (keeps effective step size sane).
- Target all attention projections plus the MLP (`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` on Llama-likes — attention-only adapters underperform measurably).
- Learning rate near 2e-4 for QLoRA with a short warmup.
- Gradient checkpointing on unless VRAM is idle.
- Batch size 4–16 with accumulation to taste (small batches are noisier but generalize fine at these scales).
- None of this substitutes for data: a clean thousand pairs beats a noisy hundred thousand, and the decider eval discipline (held-out cases, Brier and ECE alongside accuracy) applies unchanged.

## Calibration is not optional for gates

- Both Jev-class projects shipped over-confident and fixed it identically: one temperature per question type (and option count), fit on held-out calibration cases.
- Any adapter whose probabilities gate actions — tool calls, merges, human escalation — gets the same treatment, or the gate is theater.
- Probabilities you never act on can stay uncalibrated; probabilities you act on cannot.

## Learning resources

- Papers: LoRA (https://arxiv.org/abs/2106.09685), QLoRA (https://arxiv.org/abs/2305.14314), DoRA (https://arxiv.org/abs/2402.10969).
- Libraries: Hugging Face PEFT (https://github.com/huggingface/peft) for every variant above; bitsandbytes (https://github.com/bitsandbytes-foundation/bitsandbytes) for the quantization underneath; Unsloth (https://github.com/unslothai/unsloth, docs at https://docs.unsloth.ai) for the fastest practical QLoRA loop; Axolotl (https://github.com/axolotl-ai-cloud/axolotl) when configs outgrow notebooks; Liger kernels (https://github.com/linkedin/Liger-Kernel) for memory-efficient fused ops.
- Memory mechanics: PyTorch activation checkpointing (https://pytorch.org/docs/stable/checkpoint.html); FSDP when one card stops being enough (https://pytorch.org/docs/stable/fsdp.html); DeepSpeed ZeRO for the rest (https://github.com/microsoft/DeepSpeed).
- Directly applicable prior art: the Laya fine-tune notebook on free 2×T4 hardware (https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb); the open Jev reproductions at https://github.com/mihado/awesome-jev.
