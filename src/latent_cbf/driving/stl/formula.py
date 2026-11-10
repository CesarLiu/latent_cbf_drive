"""Signal temporal logic with quantitative (robustness) semantics.

Signals are a dict ``name -> Tensor[..., T]`` sampled at a fixed rate. Every
formula maps signals to a robustness trace of the same shape ``[..., T]``, where
entry ``t`` is the robustness of the formula evaluated on the suffix starting at
``t``. Positive robustness means the formula holds, negative means it is
violated, and the magnitude is the margin.

Temporal windows are given in steps, are inclusive, and are truncated at the end
of the trace (finite-trace semantics).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Union

import torch
from torch import Tensor

Signals = Dict[str, Tensor]
SignalFn = Callable[[Signals], Tensor]


class Formula:
    def robustness(self, signals: Signals) -> Tensor:
        raise NotImplementedError

    def __call__(self, signals: Signals) -> Tensor:
        return self.robustness(signals)

    def satisfied(self, signals: Signals) -> Tensor:
        """Boolean verdict at time 0 of each trace."""
        return self.robustness(signals)[..., 0] > 0

    def __and__(self, other: "Formula") -> "Formula":
        return And(self, other)

    def __or__(self, other: "Formula") -> "Formula":
        return Or(self, other)

    def __invert__(self) -> "Formula":
        return Not(self)


class Predicate(Formula):
    """Atomic proposition ``scale * (signal - threshold) > 0``.

    ``signal`` is a key of the signal dict or a callable computing a trace.
    Use ``scale=-1`` for an upper bound (``signal < threshold``).
    """

    def __init__(
        self,
        signal: Union[str, SignalFn],
        threshold: float = 0.0,
        scale: float = 1.0,
        name: Optional[str] = None,
    ) -> None:
        self.signal = signal
        self.threshold = threshold
        self.scale = scale
        self.name = name or (signal if isinstance(signal, str) else "predicate")

    def robustness(self, signals: Signals) -> Tensor:
        x = self.signal(signals) if callable(self.signal) else signals[self.signal]
        return self.scale * (x - self.threshold)

    def __repr__(self) -> str:
        op = ">" if self.scale > 0 else "<"
        return f"({self.name} {op} {self.threshold})"


def GreaterThan(signal: Union[str, SignalFn], threshold: float, scale: float = 1.0):
    return Predicate(signal, threshold, abs(scale))


def LessThan(signal: Union[str, SignalFn], threshold: float, scale: float = 1.0):
    return Predicate(signal, threshold, -abs(scale))


class And(Formula):
    def __init__(self, *children: Formula) -> None:
        self.children: Sequence[Formula] = children

    def robustness(self, signals: Signals) -> Tensor:
        return torch.stack([c(signals) for c in self.children], 0).amin(0)

    def __repr__(self) -> str:
        return "(" + " & ".join(map(repr, self.children)) + ")"


class Or(Formula):
    def __init__(self, *children: Formula) -> None:
        self.children: Sequence[Formula] = children

    def robustness(self, signals: Signals) -> Tensor:
        return torch.stack([c(signals) for c in self.children], 0).amax(0)

    def __repr__(self) -> str:
        return "(" + " | ".join(map(repr, self.children)) + ")"


class Not(Formula):
    def __init__(self, child: Formula) -> None:
        self.child = child

    def robustness(self, signals: Signals) -> Tensor:
        return -self.child(signals)

    def __repr__(self) -> str:
        return f"!{self.child!r}"


class Implies(Formula):
    def __init__(self, antecedent: Formula, consequent: Formula) -> None:
        self.antecedent = antecedent
        self.consequent = consequent

    def robustness(self, signals: Signals) -> Tensor:
        return torch.maximum(-self.antecedent(signals), self.consequent(signals))

    def __repr__(self) -> str:
        return f"({self.antecedent!r} -> {self.consequent!r})"


def _window_reduce(x: Tensor, start: int, end: Optional[int], use_min: bool) -> Tensor:
    """Reduce ``x[..., t+start : t+end+1]`` for every ``t`` with min or max.

    Windows are truncated at the end of the trace. A window that starts beyond
    the trace is empty and yields the neutral element (+inf for min, -inf for
    max), i.e. "always" holds vacuously and "eventually" fails.
    """
    T = x.shape[-1]
    neutral = float("inf") if use_min else float("-inf")
    if start < 0 or (end is not None and end < start):
        raise ValueError(f"invalid window [{start}, {end}]")
    last = T - 1 if end is None else end
    width = max(1, min(last - start + 1, T))
    pad = x.new_full((*x.shape[:-1], start + width - 1), neutral)
    padded = torch.cat([x, pad], dim=-1)[..., start:]
    windows = padded.unfold(-1, width, 1)  # [..., T, width]
    return windows.amin(-1) if use_min else windows.amax(-1)


class Always(Formula):
    """``G_[start, end] child``; ``end=None`` means until the end of the trace."""

    def __init__(self, child: Formula, start: int = 0, end: Optional[int] = None):
        self.child, self.start, self.end = child, start, end

    def robustness(self, signals: Signals) -> Tensor:
        return _window_reduce(self.child(signals), self.start, self.end, True)

    def __repr__(self) -> str:
        return f"G[{self.start},{self.end}]{self.child!r}"


class Eventually(Formula):
    """``F_[start, end] child``; ``end=None`` means until the end of the trace."""

    def __init__(self, child: Formula, start: int = 0, end: Optional[int] = None):
        self.child, self.start, self.end = child, start, end

    def robustness(self, signals: Signals) -> Tensor:
        return _window_reduce(self.child(signals), self.start, self.end, False)

    def __repr__(self) -> str:
        return f"F[{self.start},{self.end}]{self.child!r}"


class Until(Formula):
    """Unbounded ``left U right``: right eventually holds, left holds before."""

    def __init__(self, left: Formula, right: Formula) -> None:
        self.left, self.right = left, right

    def robustness(self, signals: Signals) -> Tensor:
        left, right = self.left(signals), self.right(signals)
        T = left.shape[-1]
        out = torch.empty_like(left)
        # Backward recursion: rho(t) = max(right(t), min(left(t), rho(t+1))).
        out[..., T - 1] = right[..., T - 1]
        for t in range(T - 2, -1, -1):
            out[..., t] = torch.maximum(
                right[..., t], torch.minimum(left[..., t], out[..., t + 1])
            )
        return out

    def __repr__(self) -> str:
        return f"({self.left!r} U {self.right!r})"
