# Copyright (c) OpenMRLab. All rights reserved.
"""Swap-Muon optimizer state-relocation wrapper tests.

Pure-CPU unit tests for the async copy-back wrappers introduced by the ``swap``
feature (commit b8e49e9). The wrappers drive an inner "update" generator to
completion, then copy the final device-side optimizer state back into the
pinned-CPU buffer. These tests verify the copy-back contract without a GPU or a
distributed process group.

1. TestMuonSwapWrapper  — ``_muon_swap_async_wrapper``: the final device momentum
                          is copied to CPU; ``None`` entries are skipped; an
                          empty inner does not hang; the inner is fully consumed.
2. TestAdamWSwapWrapper — ``_adamw_swap_async_wrapper``: momentum and variance
                          are both copied back, with per-buffer ``None`` skipping.
"""

from collections.abc import Generator

import torch

from xtuner.v1.optim.muon import _adamw_swap_async_wrapper, _muon_swap_async_wrapper


def _drain(generator: Generator[None, None, None]) -> None:
    """Drive ``generator`` until ``StopIteration``, mirroring ``AsyncTask``.

    Args:
        generator (Generator[None, None, None]): the wrapper generator to drain.
    """
    while True:
        try:
            next(generator)
        except StopIteration:
            return


def _make_inner(
    device_bufs: list[torch.Tensor],
    steps: list[list[torch.Tensor]],
    consumed: list[bool],
) -> Generator[None, None, None]:
    """Build a fake update generator that writes each step's values into ``device_bufs``.

    Sets ``consumed[0] = True`` once the generator is fully exhausted, so the
    test can assert the wrapper drove ``inner`` to completion.

    Args:
        device_bufs (list[torch.Tensor]): device-side buffers mutated in place per step.
        steps (list[list[torch.Tensor]]): per-step list of values, one per buffer.
        consumed (list[bool]): single-element flag set True when the generator ends.

    Yields:
        None: once per step, matching the real ``muon_update_batch_async`` cadence.
    """
    for vals in steps:
        for buf, v in zip(device_bufs, vals):
            buf.copy_(v)
        yield
    consumed[0] = True


class TestMuonSwapWrapper:
    """``_muon_swap_async_wrapper`` copy-back contract."""

    def test_copies_final_device_state_to_cpu(self):
        """The CPU buffer must hold the last value written to the device buffer."""
        cpu = [torch.zeros(4)]
        dev = [torch.zeros(4)]
        steps = [
            [torch.tensor([1.0, 2.0, 3.0, 4.0])],
            [torch.tensor([5.0, 6.0, 7.0, 8.0])],
        ]
        consumed = [False]
        inner = _make_inner(dev, steps, consumed)
        wrapper = _muon_swap_async_wrapper(inner, cpu, dev)

        _drain(wrapper)

        assert consumed[0] is True
        torch.testing.assert_close(cpu[0], torch.tensor([5.0, 6.0, 7.0, 8.0]))

    def test_multiple_buffers_copied_independently(self):
        """Each (cpu, dev) pair is copied; buffers are independent tensors."""
        cpu = [torch.zeros(2), torch.zeros(3)]
        dev = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0])]
        consumed = [False]
        inner = _make_inner(dev, [[dev[0].clone(), dev[1].clone()]], consumed)
        wrapper = _muon_swap_async_wrapper(inner, cpu, dev)

        _drain(wrapper)

        torch.testing.assert_close(cpu[0], torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(cpu[1], torch.tensor([3.0, 4.0, 5.0]))

    def test_skips_none_entries(self):
        """``None`` buffers (unused slots in a padded batch) must be skipped, not crash."""
        cpu = [torch.zeros(2), None]
        dev = [torch.tensor([9.0, 10.0]), None]
        consumed = [False]
        inner = _make_inner([dev[0]], [[torch.tensor([9.0, 10.0])]], consumed)
        wrapper = _muon_swap_async_wrapper(inner, cpu, dev)

        _drain(wrapper)  # must not raise on the None pair

        torch.testing.assert_close(cpu[0], torch.tensor([9.0, 10.0]))
        assert cpu[1] is None

    def test_empty_inner_does_not_hang(self):
        """An inner that yields nothing must still drain cleanly (no infinite loop)."""
        cpu = [torch.zeros(3)]
        dev = [torch.zeros(3)]
        consumed = [False]
        inner = _make_inner(dev, [], consumed)
        wrapper = _muon_swap_async_wrapper(inner, cpu, dev)

        _drain(wrapper)

        assert consumed[0] is True  # generator body reached its end


class TestAdamWSwapWrapper:
    """``_adamw_swap_async_wrapper`` copy-back contract (momentum + variance)."""

    def test_copies_both_momentum_and_variance(self):
        """Both the momentum and variance device buffers must be copied to CPU."""
        cpu_m = [torch.zeros(2)]
        cpu_v = [torch.zeros(2)]
        dev_m = [torch.tensor([1.0, 2.0])]
        dev_v = [torch.tensor([3.0, 4.0])]
        consumed = [False]
        inner = _make_inner(
            [dev_m[0], dev_v[0]],
            [[torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]],
            consumed,
        )
        wrapper = _adamw_swap_async_wrapper(inner, cpu_m, cpu_v, dev_m, dev_v)

        _drain(wrapper)

        torch.testing.assert_close(cpu_m[0], torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(cpu_v[0], torch.tensor([3.0, 4.0]))

    def test_skips_none_independently(self):
        """A ``None`` momentum must not block copying its (non-None) variance and vice versa."""
        # momentum slot is None, variance slot is present
        cpu_m = [None]
        cpu_v = [torch.zeros(2)]
        dev_m = [None]
        dev_v = [torch.tensor([7.0, 8.0])]
        consumed = [False]
        inner = _make_inner([dev_v[0]], [[torch.tensor([7.0, 8.0])]], consumed)
        wrapper = _adamw_swap_async_wrapper(inner, cpu_m, cpu_v, dev_m, dev_v)

        _drain(wrapper)

        torch.testing.assert_close(cpu_v[0], torch.tensor([7.0, 8.0]))
        assert cpu_m[0] is None
