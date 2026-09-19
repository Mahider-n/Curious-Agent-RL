"""CPU-friendly adapter for Craftax-Classic symbolic observations.

Craftax uses a functional JAX API. This adapter owns the PRNG key and
environment state so the existing PyTorch DQN trainers can use a conventional
``reset``/``step`` interface without changing either learning algorithm.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


CRAFTAX_CLASSIC_SYMBOLIC = "Craftax-Classic-Symbolic-v1"
EXPECTED_STATE_DIM = 1345
EXPECTED_NUM_ACTIONS = 17


def _to_host(value: Any) -> Any:
    """Convert a nested JAX result into Python scalars and NumPy arrays."""
    if isinstance(value, Mapping):
        return {key: _to_host(item) for key, item in value.items()}

    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    return array


class CraftaxClassicAdapter:
    """Stateful wrapper around ``Craftax-Classic-Symbolic-v1``.

    Imports are intentionally lazy. GridWorld users can import the package
    without installing the optional Craftax/JAX dependencies.
    """

    def __init__(
        self,
        seed: int = 0,
        max_episode_steps: int = 10_000,
        env_name: str = CRAFTAX_CLASSIC_SYMBOLIC,
    ) -> None:
        """Initialize the fixed symbolic Craftax-Classic environment."""
        if env_name != CRAFTAX_CLASSIC_SYMBOLIC:
            raise ValueError(
                "This adapter only supports "
                f"{CRAFTAX_CLASSIC_SYMBOLIC!r}; received {env_name!r}"
            )
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")

        try:
            import jax
            from craftax.craftax_env import make_craftax_env_from_name
        except ImportError as exc:
            raise ImportError(
                "Craftax support is optional. Install the local environment "
            ) from exc

        self._jax = jax
        self.env_name = env_name
        self.max_episode_steps = int(max_episode_steps)
        self._env = make_craftax_env_from_name(env_name, auto_reset=False)
        self._params = self._env.default_params.replace(
            max_timesteps=self.max_episode_steps
        )
        self._reset_fn = jax.jit(self._env.reset)
        self._step_fn = jax.jit(self._env.step)
        self._key = jax.random.PRNGKey(int(seed))
        self._state: Any | None = None

        observation_shape = tuple(
            self._env.observation_space(self._params).shape
        )
        self.state_dim = int(np.prod(observation_shape))
        self.num_actions = int(self._env.num_actions)

        if self.state_dim != EXPECTED_STATE_DIM:
            raise RuntimeError(
                "Unexpected Craftax-Classic symbolic observation size: "
                f"expected {EXPECTED_STATE_DIM}, received {self.state_dim}"
            )
        if self.num_actions != EXPECTED_NUM_ACTIONS:
            raise RuntimeError(
                "Unexpected Craftax-Classic action count: "
                f"expected {EXPECTED_NUM_ACTIONS}, received {self.num_actions}"
            )

    def _next_key(self) -> Any:
        """Advance the JAX PRNG and return a key for one operation."""
        self._key, operation_key = self._jax.random.split(self._key)
        return operation_key

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Generate a fresh procedural world and return its symbolic state."""
        if seed is not None:
            self._key = self._jax.random.PRNGKey(int(seed))

        observation, self._state = self._reset_fn(
            self._next_key(), self._params
        )
        # JAX host views may be marked read-only. PyTorch warns when wrapping
        # such arrays, so return an owned writable copy at the API boundary.
        return np.array(observation, dtype=np.float32, copy=True)

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Advance the current world without automatically resetting it."""
        if self._state is None:
            raise RuntimeError("reset() must be called before step()")
        if not 0 <= int(action) < self.num_actions:
            raise ValueError(
                f"action must be in [0, {self.num_actions}); received {action}"
            )

        observation, self._state, reward, done, info = self._step_fn(
            self._next_key(), self._state, int(action), self._params
        )
        return (
            np.array(observation, dtype=np.float32, copy=True),
            float(reward),
            bool(done),
            _to_host(info),
        )

    def close(self) -> None:
        """Match the GridWorld interface; Craftax owns no external resources."""


__all__ = [
    "CRAFTAX_CLASSIC_SYMBOLIC",
    "EXPECTED_NUM_ACTIONS",
    "EXPECTED_STATE_DIM",
    "CraftaxClassicAdapter",
]
