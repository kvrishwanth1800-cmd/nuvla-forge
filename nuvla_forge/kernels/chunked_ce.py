"""Memory-efficient cross entropy for the reasoning head.

The problem
-----------
The VQA head projects [M, H] hidden states to [M, V] logits. With M = 8192
tokens in flight and V = 32000, that tensor is ~0.5 GB in bf16. Autograd then
keeps it alive until backward *and* allocates a second one for its gradient. On
a 24 GB card that is the allocation that decides your batch size, and batch size
is throughput.

Two tricks, stacked
-------------------
1. **Chunking.** Process the M axis in slices. Only one [C, V] logit tile is ever
   live. Peak logit memory drops from O(M*V) to O(C*V).

2. **Gradient-during-forward.** For cross entropy, ``dlogits = softmax(logits) -
   onehot(target)``, which depends only on things available in forward. So each
   chunk computes its own gradient immediately, folds it into the accumulators
   for ``d_hidden`` and ``d_lm_weight``, and drops the logit tile. Nothing needs
   to survive to backward except two gradient tensors that are the size of the
   parameters, not the size of the activations.

   Backward then only has to scale by the incoming ``grad_output`` scalar, which
   is correct because mean-reduced CE is linear in the upstream gradient.

The Triton kernel
-----------------
``_ce_fwd_bwd`` does the per-chunk work in one launch with an online softmax:
a first sweep over V tracks a running max and running exponential sum with the
standard rescaling update, giving the log-sum-exp without ever holding the
exponentials; a second sweep writes ``softmax - onehot`` back **into the logit
buffer in place**. In-place is the point — it means the gradient tile costs zero
additional bytes.

Rows whose target is ``ignore_index`` contribute nothing and are zeroed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .utils import HAS_TRITON, next_power_of_2

if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _ce_fwd_bwd(
        LOGITS,      # [C, V] -- overwritten in place with dlogits
        TARGETS,     # [C]
        LOSS,        # [C] out
        stride_row,
        V,
        ignore_index,
        inv_count,   # 1 / number of non-ignored tokens in the whole batch
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0)
        target = tl.load(TARGETS + row)
        base = LOGITS + row * stride_row

        if target == ignore_index:
            tl.store(LOSS + row, 0.0)
            for start in range(0, V, BLOCK_V):
                cols = start + tl.arange(0, BLOCK_V)
                tl.store(base + cols, 0.0, mask=cols < V)
            return

        # --- sweep 1: online max and exponential sum -------------------------
        running_max = -float("inf")
        running_sum = 0.0
        for start in range(0, V, BLOCK_V):
            cols = start + tl.arange(0, BLOCK_V)
            mask = cols < V
            vals = tl.load(base + cols, mask=mask, other=-float("inf")).to(tl.float32)

            block_max = tl.max(vals, axis=0)
            new_max = tl.maximum(running_max, block_max)
            # Rescale the old sum into the new max's frame, then add this block.
            running_sum = running_sum * tl.exp(running_max - new_max) + tl.sum(
                tl.where(mask, tl.exp(vals - new_max), 0.0), axis=0
            )
            running_max = new_max

        lse = running_max + tl.log(running_sum)
        target_logit = tl.load(base + target).to(tl.float32)
        tl.store(LOSS + row, (lse - target_logit) * inv_count)

        # --- sweep 2: dlogits = (softmax - onehot) / count, written in place --
        for start in range(0, V, BLOCK_V):
            cols = start + tl.arange(0, BLOCK_V)
            mask = cols < V
            vals = tl.load(base + cols, mask=mask, other=0.0).to(tl.float32)
            probs = tl.exp(vals - lse)
            grad = tl.where(cols == target, probs - 1.0, probs) * inv_count
            tl.store(base + cols, grad, mask=mask)


def _ce_chunk_torch(logits, targets, ignore_index, inv_count):
    """Reference for one chunk. Returns (summed_loss, dlogits) already scaled."""
    valid = targets != ignore_index
    logits = logits.float()

    lse = torch.logsumexp(logits, dim=-1)
    safe_targets = targets.clamp(min=0)
    target_logit = logits.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
    loss = torch.where(valid, lse - target_logit, torch.zeros_like(lse))

    probs = torch.softmax(logits, dim=-1)
    probs.scatter_add_(
        1, safe_targets.unsqueeze(1), -torch.ones_like(target_logit).unsqueeze(1)
    )
    probs = probs * valid.unsqueeze(1)

    return loss.sum() * inv_count, probs * inv_count


class _ChunkedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, lm_weight, targets, chunk_size, ignore_index, force_reference):
        m, h_dim = hidden.shape
        v = lm_weight.shape[0]

        n_valid = int((targets != ignore_index).sum().item())
        inv_count = 1.0 / max(n_valid, 1)

        loss_total = hidden.new_zeros((), dtype=torch.float32)
        d_hidden = torch.zeros_like(hidden, dtype=torch.float32)
        d_weight = torch.zeros_like(lm_weight, dtype=torch.float32)

        use_triton = HAS_TRITON and not force_reference and hidden.is_cuda
        block_v = min(next_power_of_2(v), 4096) if use_triton else None

        for start in range(0, m, chunk_size):
            stop = min(start + chunk_size, m)
            h_chunk = hidden[start:stop]
            t_chunk = targets[start:stop]
            rows = stop - start

            # This matmul is the only place the [C, V] tile exists.
            logits = F.linear(h_chunk.float(), lm_weight.float())

            if use_triton:
                loss_buf = torch.empty(rows, dtype=torch.float32, device=hidden.device)
                _ce_fwd_bwd[(rows,)](
                    logits, t_chunk, loss_buf,
                    logits.stride(0),
                    v,
                    ignore_index,
                    inv_count,
                    BLOCK_V=block_v,
                    num_warps=8,
                )
                loss_total += loss_buf.sum()
                dlogits = logits  # overwritten in place by the kernel
            else:
                chunk_loss, dlogits = _ce_chunk_torch(
                    logits, t_chunk, ignore_index, inv_count
                )
                loss_total += chunk_loss

            d_hidden[start:stop] = dlogits @ lm_weight.float()
            d_weight += dlogits.T @ h_chunk.float()

            del logits

        ctx.save_for_backward(d_hidden, d_weight)
        ctx.hidden_dtype = hidden.dtype
        ctx.weight_dtype = lm_weight.dtype
        return loss_total

    @staticmethod
    def backward(ctx, grad_output):
        d_hidden, d_weight = ctx.saved_tensors
        g = grad_output
        return (
            (d_hidden * g).to(ctx.hidden_dtype),
            (d_weight * g).to(ctx.weight_dtype),
            None, None, None, None,
        )


def chunked_cross_entropy(
    hidden: torch.Tensor,
    lm_weight: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 1024,
    ignore_index: int = -100,
    force_reference: bool = False,
) -> torch.Tensor:
    """Mean cross entropy over ``hidden @ lm_weight.T`` without a full logit tensor.

    Args:
        hidden: [M, H] pre-projection hidden states.
        lm_weight: [V, H] output embedding.
        targets: [M] int64 class indices, ``ignore_index`` to mask.
        chunk_size: rows of M per tile. Bigger is faster and uses more memory;
            1024 is a reasonable default for V around 32k on a 24 GB card.
        ignore_index: target value that contributes no loss and no gradient.

    Returns:
        Scalar loss, mean-reduced over non-ignored tokens.
    """
    return _ChunkedCrossEntropy.apply(
        hidden, lm_weight, targets, chunk_size, ignore_index, force_reference
    )
