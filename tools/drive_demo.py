"""Drive a bridge bot through a sequence of compass hops so the
prismarine-viewer renders a live trajectory. Standalone demo helper —
not part of the eval pipeline.

    python3 tools/drive_demo.py [n_hops=12] [port=9000]
"""
import json
import socket
import sys
import time

n_hops = int(sys.argv[1]) if len(sys.argv) > 1 else 12
port = int(sys.argv[2]) if len(sys.argv) > 2 else 9000

# A roughly outward-spiraling tour so the trail is visibly non-degenerate.
thetas = [0, 45, 90, 90, 135, 180, 180, 225, 270, 270, 315, 0]

s = socket.create_connection(("localhost", port), timeout=10)
s.settimeout(120)  # a 50-block pathfinder hop can take ~33s; allow margin
f = s.makefile("rw")


def step(theta, distance):
    f.write(json.dumps({"theta": theta, "distance": distance}) + "\n")
    f.flush()
    return json.loads(f.readline())


visited = set()
for i in range(n_hops):
    theta = thetas[i % len(thetas)]
    obs = step(theta, 50)
    visited |= set(obs.get("visitedBiomes", []))
    tag = "STUCK" if obs.get("stuck") else "ok"
    print(f"hop {i:2d} θ={theta:3d}  pos=({obs.get('x')},{obs.get('z')})  "
          f"biome={obs.get('biomeName')}  visited={len(visited)}  {tag}",
          flush=True)
    if obs.get("dead"):
        print("bot died:", obs.get("reason"))
        break
    time.sleep(0.3)

print(f"\ndone — {len(visited)} unique biomes over {n_hops} hops")
s.close()
