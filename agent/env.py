"""TCP client for the mineflayer bridge.

Each bot exposes a newline-delimited JSON socket on port 9000+id.
Request:  {"theta": <deg>, "distance": <blocks>}
Response: {"biomeId", "biomeName", "cellX", "cellZ", "x", "z",
           "health", "food", "numVisited", "visitedBiomes",
           "gridRadius", "grid", "stuck"?}

Action space: 8-way compass at a fixed distance.

If constructed with a `WorldView` (complete-knowledge setting, proposal
§2), the bridge's `grid` field is replaced by a slice from the view
before the obs is returned. In line-of-sight mode (no view passed) the
bridge's grid is forwarded as-is.
"""

import json
import socket

from agent.world import WorldView

NUM_ACTIONS = 8
DEFAULT_DISTANCE = 100
DEFAULT_GRID_RADIUS = 8  # cells; must match bot/bridge.js GRID_RADIUS default


def action_to_theta(action: int) -> float:
    if not 0 <= action < NUM_ACTIONS:
        raise ValueError(f"action {action} out of range [0, {NUM_ACTIONS})")
    return action * (360.0 / NUM_ACTIONS)


class Env:
    def __init__(self, host: str = "localhost", port: int = 9000,
                 distance: int = DEFAULT_DISTANCE, timeout: float = 120.0,
                 world_view: WorldView | None = None,
                 grid_radius: int = DEFAULT_GRID_RADIUS):
        self.distance = distance
        self.world_view = world_view
        self.grid_radius = grid_radius
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.settimeout(timeout)
        self._buf = b""

    def step(self, action: int) -> dict:
        msg = {"theta": action_to_theta(action), "distance": self.distance}
        return self._send(msg)

    def _send(self, msg: dict) -> dict:
        self.sock.sendall((json.dumps(msg) + "\n").encode())
        obs = self._read_line()
        if self.world_view is not None:
            grid = self.world_view.get_grid(obs["cellX"], obs["cellZ"], self.grid_radius)
            obs["grid"] = grid.flatten().tolist()
            obs["gridRadius"] = self.grid_radius
        return obs

    def _read_line(self) -> dict:
        while b"\n" not in self._buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("bridge closed")
            self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\n")
        return json.loads(line)

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass
