# neural.py

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
        self._activation_clip = 50.0
        self._grad_clip = 5.0
        self._weight_clip = 10.0

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
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=10.0, neginf=-10.0)
        return normalized.astype(np.float32)

    # ------------------------------------------------------------------
    # forward/backward helpers
    def _forward(self, features: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        activations = [features]
        pre_acts = []
        for i, layer in enumerate(self._layers):
            z = np.dot(activations[-1], layer.weight) + layer.bias
            pre_acts.append(z)
            if i < len(self._layers) - 1:
                a = np.maximum(z, 0.0)  # ReLU for hidden layers
                a = np.clip(a, -self._activation_clip, self._activation_clip)
            else:
                a = z  # Linear output layer
            activations.append(a)
        return activations, pre_acts

    def predict(self, features: np.ndarray) -> np.ndarray:
        if features.size == 0:
            return np.array([])
        normalized = self._normalise_features(features)
        activations, _ = self._forward(normalized)
        return activations[-1].squeeze(-1).astype(np.float32)

    def update(self, features: np.ndarray, rewards: np.ndarray) -> None:
        if features.shape[0] != rewards.shape[0] or features.shape[0] < 2:
            return  # Need at least two samples for contrastive loss

        normalized = self._normalise_features(features)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1, 1)

        for epoch in range(self.max_epochs):
            # Forward pass
            activations, pre_acts = self._forward(normalized)

            # Pairwise contrastive loss
            preds = activations[-1]
            n = preds.shape[0]
            diff = preds - preds.T            # (n, n) pred differences
            label_diff = (rewards - rewards.T > 0).astype(np.float32)  # (n, n) bool mask
            loss = -np.mean(diff * label_diff)

            # Add L2 regularization
            reg_loss = 0.0
            for layer in self._layers:
                reg_loss += np.sum(layer.weight ** 2)
            loss += self.weight_decay * reg_loss

            if np.isnan(loss) or np.isinf(loss):
                logger.warning("NaN/Inf loss detected; skipping update")
                return

            # Backward pass — proper contrastive gradient
            # d(loss)/d(pred_k) = -1/n^2 * (sum_j label(k>j) - sum_j label(j>k))
            # i.e. positive when k should rank higher, negative otherwise
            grad_pred = -(label_diff.sum(axis=1, keepdims=True)
                          - label_diff.sum(axis=0, keepdims=True).T) / (n * n)
            upstream = grad_pred
            grads_w = []
            grads_b = []

            for layer_index in range(len(self._layers) - 1, -1, -1):
                layer = self._layers[layer_index]
                input_act = activations[layer_index]

                grad_w = np.dot(input_act.T, upstream)
                grad_b = np.sum(upstream, axis=0)

                grads_w.insert(0, grad_w)
                grads_b.insert(0, grad_b)

                if layer_index > 0:
                    upstream = np.dot(upstream, layer.weight.T)
                    upstream = np.clip(
                        upstream, -self._grad_clip, self._grad_clip
                    )
                    relu_grad = (pre_acts[layer_index - 1] > 0).astype(np.float32)
                    upstream = upstream * relu_grad
                    upstream = np.clip(upstream, -self._grad_clip, self._grad_clip)

            # Update weights with momentum
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
                layer.weight = np.clip(
                    layer.weight, -self._weight_clip, self._weight_clip
                ).astype(np.float32)
                layer.bias = np.clip(
                    layer.bias, -self._weight_clip, self._weight_clip
                ).astype(np.float32)

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