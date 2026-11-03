"""Synthetic correctness test for the signed full-vocabulary Outcome-PI loss (§8).

Objective under test:
    L = - sum_v stopgrad(d_v) * log_softmax(z)_v
with d = (2R-1) * (q - p_S), q a valid distribution, so sum_v d_v = 0.

Analytic gradient (using sum_v d_v = 0):
    dL/dz_v = -d_v
The test checks, with autograd, both the success (R=1) and failure (R=0)
directions, that dL/dz ~= -d, and that sum(d) ~= 0.

CPU-only, no framework. Run: python tests/test_signed_loss.py
"""

import torch

V = 1000
T = 7
torch.manual_seed(0)


def signed_loss(z: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """L = -sum_v stopgrad(d_v) * log_softmax(z)_v  (per-position sum over vocab)."""
    return -(d.detach() * torch.log_softmax(z, dim=-1)).sum(dim=-1)


def check_direction(tag: str, d_builder) -> None:
    z = torch.randn(T, V, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        p_s = torch.softmax(z, dim=-1)
        q_logits = torch.randn(T, V, dtype=torch.float64)
        q = torch.softmax(q_logits, dim=-1)
        d = d_builder(p_s, q)

    assert d.shape == (T, V)
    assert torch.allclose(d.sum(dim=-1), torch.zeros(T, dtype=torch.float64), atol=1e-10), (
        f"{tag}: sum(d) != 0"
    )

    loss = signed_loss(z, d).sum()
    loss.backward()
    grad = z.grad

    expected = -d
    max_abs = (grad - expected).abs().max().item()
    rel = max_abs / expected.abs().max().item()
    print(f"{tag}: sum(d)={d.sum().item():+.2e}  max|dL/dz - (-d)|={max_abs:.3e} (rel {rel:.2e})")
    assert max_abs < 1e-10, f"{tag}: gradient mismatch {max_abs}"


def main() -> None:
    check_direction("success R=1: d = q - p_S", lambda p_s, q: q - p_s)
    check_direction("failure R=0: d = p_S - q", lambda p_s, q: p_s - q)

    # bf16-tolerance variant (training runs in bf16)
    z = torch.randn(T, V, dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        p_s = torch.softmax(z, dim=-1)
        q = torch.softmax(torch.randn(T, V), dim=-1)
        d = (q - p_s).to(torch.bfloat16).float()
    loss = signed_loss(z, d).sum()
    loss.backward()
    max_abs = (z.grad - (-d)).abs().max().item()
    print(f"bf16 stopgrad d: max|dL/dz - (-d)|={max_abs:.3e} (bf16-tolerant)")
    assert max_abs < 1e-4

    print("SIGNED LOSS UNIT TEST: PASS")


if __name__ == "__main__":
    main()
