import numpy as np

from umbra.neural import NeuralRewardModel


def test_neural_reward_model_handles_large_batches() -> None:
    model = NeuralRewardModel()
    features = np.full((64, model.input_dim), 1e6, dtype=np.float32)
    rewards = np.full((64,), 5e3, dtype=np.float32)

    model.update(features, rewards)

    prediction = model.predict(features[0])
    assert np.isfinite(prediction)
