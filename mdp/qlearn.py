"""Linear Q-learning with ε-greedy (proposal §6).

  Q(s, a) = w_a · φ(s)
  TD(0): δ = r + γ · max_a' Q(s', a') - Q(s, a)
         w_a += α · δ · φ(s)

Per-action weight vectors share the state featurizer φ(s) from
mdp/features.py. With 8 actions and PHI_DIM = 17, the model has
8 × 17 = 136 parameters — well-matched to our ~360–750 action training
budget.
"""

import numpy as np
from pathlib import Path

from mdp.env import NUM_ACTIONS
from mdp.features import PHI_DIM, featurize


class LinearQ:
    def __init__(self, *, alpha: float = 0.05, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_min: float = 0.05,
                 epsilon_decay_episodes: int = 20, seed: int | None = None):
        self.W = np.zeros((NUM_ACTIONS, PHI_DIM), dtype=np.float32)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        # Weights persist across episodes (that's the whole point of training);
        # nothing to reset here. Method exists for Policy-protocol compatibility.
        pass

    def q_values(self, obs: dict) -> np.ndarray:
        return self.W @ featurize(obs)

    def act(self, obs: dict) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(NUM_ACTIONS))
        return int(np.argmax(self.q_values(obs)))

    def update(self, s: dict, a: int, r: float, s_next: dict, done: bool) -> float:
        """TD(0) update; returns the TD error for logging."""
        phi_s = featurize(s)
        if done:
            target = r
        else:
            target = r + self.gamma * float(np.max(self.W @ featurize(s_next)))
        td_error = target - float(self.W[a] @ phi_s)
        self.W[a] += self.alpha * td_error * phi_s
        return td_error

    def decay_epsilon(self, episode_idx: int) -> None:
        """Linear decay from 1.0 to epsilon_min over the first N episodes."""
        if episode_idx >= self.epsilon_decay_episodes:
            self.epsilon = self.epsilon_min
        else:
            frac = episode_idx / self.epsilon_decay_episodes
            self.epsilon = max(self.epsilon_min, 1.0 - (1.0 - self.epsilon_min) * frac)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, W=self.W,
                 alpha=np.float32(self.alpha),
                 gamma=np.float32(self.gamma),
                 epsilon=np.float32(self.epsilon))

    @classmethod
    def load(cls, path: Path) -> "LinearQ":
        z = np.load(path)
        agent = cls(alpha=float(z["alpha"]), gamma=float(z["gamma"]),
                    epsilon=float(z["epsilon"]))
        agent.W = z["W"].astype(np.float32)
        return agent


def compute_reward(prev_obs: dict, obs: dict) -> float:
    """Reward function (milestone): +1 per new biome, -0.1 stuck, -0.01 step."""
    delta = int(obs.get("numVisited", 0)) - int(prev_obs.get("numVisited", 0))
    r = float(delta)
    if obs.get("stuck"):
        r -= 0.1
    r -= 0.01
    return r
