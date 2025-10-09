"""Neural helpers for adaptive reward modelling in Project Umbra."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class _LayerState:
    weight: np.ndarray
    bias: np.ndarray
    velocity_w: np.ndarray
    velocity_b: np.ndarray


class NeuralRewardModel:
    """A lightweight multilayer perceptron used to rank evolution candidates.

    The model is intentionally small so it can be trained on-the-fly using the
    metrics gathered from each generation. It operates purely with ``numpy``
    primitives which keeps it lightweight and avoids heavy framework
    dependencies while still providing a non-linear function approximator that
    can reward nuanced improvements.
    """

    def __init__(
        self,
        *,
        input_dim: int = 5,
        hidden_layers: Iterable[int] = (64, 32, 16),
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        max_epochs: int = 6,
    ) -> None:
        self.input_dim = int(input_dim)
        self.hidden_layers = tuple(int(width) for width in hidden_layers)
        self.learning_rate = float(learning_rate)
        self.momentum = float(np.clip(momentum, 0.0, 0.999))
        self.weight_decay = float(max(weight_decay, 0.0))
        self.max_epochs = max(1, int(max_epochs))
        self._epsilon = 1e-6
        self._init_network()
        self._feature_mean = np.zeros(self.input_dim, dtype=np.float32)
        self._feature_var = np.ones(self.input_dim, dtype=np.float32)
        self._samples_seen = 0

    def _init_network(self) -> None:
        layer_dims = (self.input_dim, *self.hidden_layers, 1)
        self._layers: list[_LayerState] = []
        rng = np.random.default_rng(12345)
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            limit = np.sqrt(6.0 / (in_dim + out_dim))
            weight = rng.uniform(-limit, limit, size=(in_dim, out_dim)).astype(np.float32)
            bias = np.zeros(out_dim, dtype=np.float32)
            self._layers.append(
                _LayerState(
                    weight=weight,
                    bias=bias,
                    velocity_w=np.zeros_like(weight),
                    velocity_b=np.zeros_like(bias),
                )
            )

    # ------------------------------------------------------------------
    # normalisation utilities
    def _normalise_features(self, features: np.ndarray) -> np.ndarray:
        if features.ndim == 1:
            features = features[None, :]
        if features.shape[1] != self.input_dim:
            raise ValueError("Feature dimension mismatch for neural reward model")

        batch_mean = features.mean(axis=0)
        batch_var = features.var(axis=0) + self._epsilon
        total_samples = self._samples_seen + features.shape[0]
        weight_new = features.shape[0] / max(total_samples, 1)
        weight_old = 1.0 - weight_new
        self._feature_mean = (
            weight_old * self._feature_mean + weight_new * batch_mean.astype(np.float32)
        )
        self._feature_var = (
            weight_old * self._feature_var + weight_new * batch_var.astype(np.float32)
        )
        self._feature_var = np.clip(self._feature_var, self._epsilon, None)
        self._samples_seen = total_samples

        normalized = (features - self._feature_mean) / np.sqrt(self._feature_var)
        return normalized.astype(np.float32)

    # ------------------------------------------------------------------
    # forward/backward helpers
    def _forward(self, features: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        activations = [features]
        pre_activations: list[np.ndarray] = []
        for idx, layer in enumerate(self._layers):
            z = activations[-1] @ layer.weight + layer.bias
            pre_activations.append(z)
            if idx == len(self._layers) - 1:
                activations.append(z)
            else:
                activations.append(np.maximum(z, 0.0).astype(np.float32))
        return activations, pre_activations

    # ------------------------------------------------------------------
    def predict(self, features: np.ndarray) -> float:
        """Return the scalar reward prediction for ``features``."""

        features = np.asarray(features, dtype=np.float32)
        norm = (features - self._feature_mean) / np.sqrt(self._feature_var)
        activations, _ = self._forward(norm.reshape(1, -1))
        return float(activations[-1].ravel()[0])

    # ------------------------------------------------------------------
    def update(self, features: np.ndarray, rewards: np.ndarray) -> None:
        """Perform a mini training loop on the provided feature/reward pairs."""

        features = np.asarray(features, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1, 1)
        if features.size == 0 or rewards.size == 0:
            return
        if features.shape[0] != rewards.shape[0]:
            raise ValueError("Feature and reward batches must share the same length")

        norm_features = self._normalise_features(features)
        for epoch in range(self.max_epochs):
            activations, pre_acts = self._forward(norm_features)
            predictions = activations[-1]
            error = predictions - rewards
            loss = float(np.mean(error**2))
            if epoch == self.max_epochs - 1:
                logger.debug("Neural reward epoch %d loss %.5f", epoch, loss)

            grad_output = (2.0 / rewards.shape[0]) * error
            grads_w: list[np.ndarray] = [np.zeros_like(layer.weight) for layer in self._layers]
            grads_b: list[np.ndarray] = [np.zeros_like(layer.bias) for layer in self._layers]
            backprop = grad_output

            for layer_index in reversed(range(len(self._layers))):
                layer = self._layers[layer_index]
                input_activation = activations[layer_index]
                grads_w[layer_index] = input_activation.T @ backprop + self.weight_decay * layer.weight
                grads_b[layer_index] = backprop.sum(axis=0)

                if layer_index:
                    upstream = backprop @ layer.weight.T
                    relu_grad = (pre_acts[layer_index - 1] > 0).astype(np.float32)
                    backprop = upstream * relu_grad

            for idx, layer in enumerate(self._layers):
                layer.velocity_w = (
                    self.momentum * layer.velocity_w
                    - self.learning_rate * grads_w[idx].astype(np.float32)
                )
                layer.velocity_b = (
                    self.momentum * layer.velocity_b
                    - self.learning_rate * grads_b[idx].astype(np.float32)
                )
                layer.weight += layer.velocity_w
                layer.bias += layer.velocity_b

    # ------------------------------------------------------------------
    def to_state(self) -> dict[str, object]:
        """Return a JSON-serialisable snapshot of the model."""

        return {
            "input_dim": self.input_dim,
            "hidden_layers": list(self.hidden_layers),
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "max_epochs": self.max_epochs,
            "layers": [
                {
                    "weight": layer.weight.astype(np.float32).tolist(),
                    "bias": layer.bias.astype(np.float32).tolist(),
                    "velocity_w": layer.velocity_w.astype(np.float32).tolist(),
                    "velocity_b": layer.velocity_b.astype(np.float32).tolist(),
                }
                for layer in self._layers
            ],
            "feature_mean": self._feature_mean.astype(np.float32).tolist(),
            "feature_var": self._feature_var.astype(np.float32).tolist(),
            "samples_seen": int(self._samples_seen),
        }

    # ------------------------------------------------------------------
    @classmethod
    def from_state(cls, state: dict[str, object]) -> NeuralRewardModel:
        """Instantiate a model from :meth:`to_state` output."""

        model = cls(
            input_dim=int(state.get("input_dim", 5)),
            hidden_layers=state.get("hidden_layers", (64, 32, 16)),
            learning_rate=float(state.get("learning_rate", 0.01)),
            momentum=float(state.get("momentum", 0.9)),
            weight_decay=float(state.get("weight_decay", 1e-4)),
            max_epochs=int(state.get("max_epochs", 6)),
        )

        layers = state.get("layers")
        if isinstance(layers, list) and len(layers) == len(model._layers):
            for stored, layer in zip(layers, model._layers):
                layer.weight = np.asarray(stored.get("weight", layer.weight), dtype=np.float32)
                layer.bias = np.asarray(stored.get("bias", layer.bias), dtype=np.float32)
                layer.velocity_w = np.asarray(
                    stored.get("velocity_w", layer.velocity_w), dtype=np.float32
                )
                layer.velocity_b = np.asarray(
                    stored.get("velocity_b", layer.velocity_b), dtype=np.float32
                )

        mean = state.get("feature_mean")
        var = state.get("feature_var")
        if mean is not None:
            model._feature_mean = np.asarray(mean, dtype=np.float32)
        if var is not None:
            model._feature_var = np.clip(
                np.asarray(var, dtype=np.float32), model._epsilon, None
            )
        model._samples_seen = int(state.get("samples_seen", 0))
        return model


__all__ = ["NeuralRewardModel"]

