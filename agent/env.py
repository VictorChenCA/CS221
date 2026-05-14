"""TCP client for the mineflayer bridge.

Each bot exposes a newline-delimited JSON socket on port 9000+id.
Request:  {"theta": <deg>, "distance": <blocks>}
Response: {"biomeId", "biomeName", "x", "z", "health", "food",
           "numVisited", "visitedBiomes", "stuck"?}

Action space: 8-way compass at a fixed distance.
"""

import json
import socket

NUM_ACTIONS = 8
DEFAULT_DISTANCE = 100


def action_to_theta(action: int) -> float:
    if not 0 <= action < NUM_ACTIONS:
        raise ValueError(f"action {action} out of range [0, {NUM_ACTIONS})")
    return action * (360.0 / NUM_ACTIONS)


class Env:
    def __init__(self, host: str = "localhost", port: int = 9000,
                 distance: int = DEFAULT_DISTANCE, timeout: float = 120.0):
        self.distance = distance
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.settimeout(timeout)
        self._buf = b""

    def step(self, action: int) -> dict:
        msg = {"theta": action_to_theta(action), "distance": self.distance}
        self.sock.sendall((json.dumps(msg) + "\n").encode())
        return self._read_line()

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
