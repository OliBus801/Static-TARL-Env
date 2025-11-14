"""Simple Q-learning agent.

This module exposes :class:`QLearningAgent`, a small utility class designed to be
plugged into reinforcement learning training loops.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Hashable, Optional

import numpy as np

try:  # Gymnasium is an optional dependency for the type hint.
    from gymnasium import spaces
except ImportError:  # pragma: no cover - gymnasium is optional at runtime.
    spaces = Any  # type: ignore


class QLearningAgent:
    """Tabular Q-learning implementation with epsilon-greedy exploration.

    Parameters
    ----------
    action_space:
        Gymnasium action space. Must expose ``n`` (number of discrete actions)
        and ``sample()`` (to draw random actions).
    state_space_size:
        Optional integer describing the cardinality of the discrete state
        space. When not provided, states are expected to be hashable and are
        stored lazily in a ``defaultdict``.
    learning_rate, discount:
        Standard Q-learning hyperparameters ``α`` and ``γ``.
    initial_epsilon, final_epsilon, epsilon_decay_rate:
        Exploration parameters. ``epsilon_decay`` should be called at the end of
        every episode (or step) to progressively anneal ``ε``.

    Attributes
    ----------
    q_table:
        Underlying Q-values storage. When ``state_space_size`` is provided it is
        an ``np.ndarray`` of shape ``(state_space_size, action_space.n)``.
        Otherwise it is a ``defaultdict`` returning zero-valued numpy arrays.
    epsilon:
        Current exploration rate used by :meth:`get_action`.

    Notes
    -----
    The agent is intentionally lightweight so that a training script can wire
    it up to almost any Gymnasium-compatible environment.
    """

    def __init__(
        self,
        action_space: "spaces.Discrete",
        state_space_size: Optional[int] = None,
        *,
        learning_rate: float = 0.1,
        discount: float = 0.99,
        initial_epsilon: float = 1.0,
        final_epsilon: float = 0.05,
        epsilon_decay_rate: float = 0.995,
    ) -> None:
        if not hasattr(action_space, "n"):
            msg = "QLearningAgent expects a discrete action space exposing `n`."
            raise ValueError(msg)

        self.action_space = action_space
        self.state_space_size = state_space_size
        self.learning_rate = learning_rate
        self.discount = discount
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon = initial_epsilon

        if state_space_size is not None:
            self.q_table: Any = np.zeros(
                (state_space_size, self.action_space.n), dtype=np.float32
            )
        else:
            self.q_table = defaultdict(self._zero_action_values)

    def _zero_action_values(self) -> np.ndarray:
        return np.zeros(self.action_space.n, dtype=np.float32)

    def _get_state_index(self, state: Hashable) -> int:
        if self.state_space_size is None:
            raise RuntimeError("State indexing is only available for tabular mode.")
        index = int(state)
        if index < 0 or index >= self.state_space_size:
            msg = (
                "State index out of bounds for the configured state space size:"
                f" {index} not in [0, {self.state_space_size})."
            )
            raise IndexError(msg)
        return index

    def _get_q_values(self, state: Hashable) -> np.ndarray:
        if self.state_space_size is not None:
            return self.q_table[self._get_state_index(state)]
        return self.q_table[state]

    def get_action(self, state: Hashable) -> int:
        """Return an epsilon-greedy action for ``state``."""

        if np.random.random() < self.epsilon:
            return int(self.action_space.sample())
        q_values = self._get_q_values(state)
        return int(np.argmax(q_values))

    def update(
        self,
        state: Hashable,
        action: int,
        reward: float,
        next_state: Hashable,
        done: bool,
    ) -> None:
        """Apply the tabular Q-learning update rule."""

        q_values = self._get_q_values(state)
        best_next = 0.0
        if not done:
            best_next = float(np.max(self._get_q_values(next_state)))
        target = reward + self.discount * best_next
        q_values[action] = (1 - self.learning_rate) * q_values[action] + self.learning_rate * target

    def epsilon_decay(self) -> None:
        """Decay ``epsilon`` while keeping it above ``final_epsilon``."""

        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay_rate)
