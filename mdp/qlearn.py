"""Linear Q-learning with ε-greedy (proposal §6).

  Q(s, a) = w_a · φ(s)
  TD(0): δ = r + γ · max_a' Q(s', a') - Q(s, a)
         w_a += α · δ · φ(s)

Per-action weight vectors share the state featurizer φ(s) from
mdp/features.py. With 8 actions and PHI_DIM = 17, the model has
8 × 17 = 136 parameters — well-matched to our ~360–750 action training
budget.
"""

import os
import numpy as np
from pathlib import Path

from mdp.env import NUM_ACTIONS
from mdp.features import PHI_DIM, featurize, novelty_potential

# Potential-based shaping weight (Ng/Harada/Russell 1999). 0 disables it.
# F(s,a,s') = β·(γ·Φ(s') − Φ(s)) with Φ = closeness to nearest novel biome,
# so the agent gets a dense per-step gradient toward novelty instead of
# only the sparse +1 on entering a new biome. Policy-invariant in the
# limit; with linear FA + little data it mainly speeds directed
# exploration. γ matches the learner's discount.
SHAPE_BETA = float(os.environ.get("SHAPE_BETA", "0.5"))
SHAPE_GAMMA = float(os.environ.get("SHAPE_GAMMA", "0.95"))


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
    """Reward shaping with stuck/unstuck bonuses (iter 13).

      +1.0  per new biome discovered (primary)
      +0.20 got-unstuck bonus: previous was stuck, current is not
      -0.03 per stuck step (existing)
      -0.005 per step (existing tiny time penalty)

    The got-unstuck bonus rewards transitioning from a stuck state to a
    moving state. With the was_stuck feature in φ(s), Q-learning can
    now learn 'pick a direction that gets me unstuck' rather than
    relying on the hardcoded eval-side stuck-escape. Penalties for
    each stuck step still apply (so loops are bad), but the +0.20
    bonus makes "trying a different direction" actively rewarding.
    """
    delta = int(obs.get("numVisited", 0)) - int(prev_obs.get("numVisited", 0))
    r = float(delta)
    if obs.get("stuck"):
        r -= 0.03
    elif prev_obs.get("stuck"):
        # was stuck, now isn't — reward for escaping
        r += 0.20
    r -= 0.005
    # Potential-based shaping toward novelty (dense exploration gradient).
    if SHAPE_BETA:
        r += SHAPE_BETA * (SHAPE_GAMMA * novelty_potential(obs)
                           - novelty_potential(prev_obs))
    return r
