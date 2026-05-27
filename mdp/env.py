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
import os
import socket
import time

from mdp.world import NpzWorldView

NUM_ACTIONS = 8
# Env-configurable so we can sweep hop distances without code edits.
DEFAULT_DISTANCE = int(os.environ.get("HOP_DISTANCE", "50"))
DEFAULT_GRID_RADIUS = 128  # cells = 32 chunks (1 chunk = 16 blocks = 4 cells)
# How long to keep retrying ConnectionRefused before giving up. The bridge
# only opens its TCP listener after the bot has spawned + dispersed +
# landed, which can take >60s on cold-boot servers while chunk gen runs.
CONNECT_RETRY_S = 120.0
CONNECT_RETRY_DELAY_S = 2.0


def action_to_theta(action: int) -> float:
    if not 0 <= action < NUM_ACTIONS:
        raise ValueError(f"action {action} out of range [0, {NUM_ACTIONS})")
    return action * (360.0 / NUM_ACTIONS)


class Env:
    def __init__(self, host: str = "localhost", port: int = 9000,
                 distance: int = DEFAULT_DISTANCE, timeout: float = 120.0,
                 world_view: NpzWorldView | None = None,
                 grid_radius: int = DEFAULT_GRID_RADIUS):
        self.distance = distance
        self.world_view = world_view
        self.grid_radius = grid_radius
        self.sock = _connect_with_retry(host, port, timeout)
        self.sock.settimeout(timeout)
        self._buf = b""

    def step(self, action: int) -> dict:
        return self.step_raw(action_to_theta(action), self.distance)

    def observe(self) -> dict:
        """Read the current obs without moving the bot.

        The bridge treats `distance=0` as a no-op that just runs `getObs`
        and returns immediately — so this skips the ~5–30s pathfinder run
        a regular `step` would burn. Useful for episode warmup."""
        return self.step_raw(0.0, 0)

    def step_raw(self, theta_deg: float, distance_blocks: int) -> dict:
        """Send a single (theta, distance) hop. Used by the oracle path,
        which carries exact distances rather than compass indices."""
        self.sock.sendall((json.dumps(
            {"theta": theta_deg, "distance": distance_blocks}) + "\n").encode())
        obs = self._read_line()
        # Two reasons env owns the world overlay rather than the bridge:
        # 1) bot.entity can be momentarily null (bot kicked/respawning);
        #    the bridge encodes NaN coords as JSON null.
        # 2) mineflayer's bot.world.getBiome() doesn't agree with Paper's
        #    1.20.1 biome registry (it returns 0 everywhere), so the
        #    bridge's biomeId field is unusable. We override it with the
        #    cubiomes-derived id from the WorldView, which is the source
        #    of truth for the whole "complete knowledge" pipeline.
        if self.world_view is not None:
            size = 2 * self.grid_radius + 1
            if obs.get("cellX") is not None and obs.get("cellZ") is not None:
                grid = self.world_view.get_grid(
                    obs["cellX"], obs["cellZ"], self.grid_radius)
                obs["grid"] = grid.flatten().tolist()
                obs["biomeId"] = self.world_view.biome_at(
                    obs["cellX"], obs["cellZ"])
            else:
                obs["grid"] = [-1] * (size * size)
                # leave biomeId as whatever bridge sent; eval skips < 0
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


def _connect_with_retry(host: str, port: int, sock_timeout: float) -> socket.socket:
    """Retry on ConnectionRefused (bridge not yet listening) but propagate
    other errors immediately. Cap total wait at CONNECT_RETRY_S."""
    deadline = time.monotonic() + CONNECT_RETRY_S
    last_err: Exception | None = None
    while True:
        try:
            return socket.create_connection((host, port), timeout=sock_timeout)
        except ConnectionRefusedError as e:
            last_err = e
            if time.monotonic() >= deadline:
                raise
            time.sleep(CONNECT_RETRY_DELAY_S)
