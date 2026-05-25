// One mineflayer bot per process. Listens on TCP port 9000+id, speaks
// newline-delimited JSON to the Python agent.
//   request:  {"theta": 90, "distance": 100}
//   response: {"biomeId", "biomeName", "cellX", "cellZ", "x", "z",
//              "health", "food", "numVisited", "visitedBiomes",
//              "stuck"?, "gridRadius"?, "grid"?}
//
// `grid`/`gridRadius` are only shipped when WORLD_MODE=los. In complete
// mode the agent overlays the grid from a pre-extracted seed dump on
// the Python side. The grid is a flat (2r+1)x(2r+1) row-major array of
// biome ids sampled on a 4-block stride, with +dz as the outer index.
// A value of -1 means "unknown": either an un-streamed chunk (the_void
// sentinel) or filtered by the visible() predicate for line-of-sight.

const net = require('net');
const mineflayer = require('mineflayer');
const { pathfinder, Movements, goals: { GoalNearXZ } } = require('mineflayer-pathfinder');
const mcData = require('minecraft-data')('1.20.1');
const { Vec3 } = require('vec3');

const ID = parseInt(process.argv[2] || '0', 10);
const HOST = process.env.MC_HOST || 'localhost';
const PORT = parseInt(process.env.MC_PORT || '25565', 10);
const BRIDGE_PORT = 9000 + ID;
const GRID_RADIUS = parseInt(process.env.GRID_RADIUS || '8', 10);  // cells (1 cell = 4 blocks)
const SPAWN_CHUNK_WAIT_MS = 2000;  // let initial chunk stream settle before serving the first obs
// Proposal §6: spawn bots on a circle so they don't pathfinder-collide
// at (0,0). DISPERSE_R = circle radius in blocks; DISPERSE_N = how many
// bots are spread around this server (used for the angle).
// 250 blocks ≈ 16 chunks; inside the server's 24-chunk view distance
// from spawn, so chunks at the tp destination are already loaded.
const DISPERSE_R = parseInt(process.env.DISPERSE_R || '250', 10);
const DISPERSE_N = parseInt(process.env.DISPERSE_N || '5', 10);
const LAND_TIMEOUT_MS = 20000;  // give the bot at most this long to land after /tp
// 'complete' = Python overlays the grid from a pre-extracted seed dump,
// so we skip live sampling here. 'los' = bridge ships its loaded-chunk
// grid (with the visible() filter) for the line-of-sight setting.
const WORLD_MODE = process.env.WORLD_MODE || 'complete';

// Visibility predicate — always-true under complete-knowledge.
// Swap this out for raycasting / heightmap checks to get line-of-sight.
function visible(_bx, _bz) { return true; }

const bot = mineflayer.createBot({
  host: HOST,
  port: PORT,
  username: `Explorer_${ID}`,
  auth: 'offline',
  version: '1.20.1',
});

bot.loadPlugin(pathfinder);
bot.visitedBiomes = new Set();

bot.once('spawn', () => {
  // canDig=true lets the bot break leaves/wood to escape canopy if it
  // lands on a tree after the Y=250 disperse fall.
  const moves = new Movements(bot);
  moves.canDig = true;
  moves.maxDropDown = 8;
  bot.pathfinder.setMovements(moves);
  // Disperse on a DISPERSE_R-block circle (proposal §6). The /tp
  // command needs op — see tools/run_test_eval.py which writes ops.json.
  const localId = ID % DISPERSE_N;
  const angle = (localId / DISPERSE_N) * 2 * Math.PI;
  const tx = Math.round(Math.cos(angle) * DISPERSE_R);
  const tz = Math.round(Math.sin(angle) * DISPERSE_R);
  console.log(`bot ${ID} spawned, dispersing to (${tx}, ${tz})`);
  // Disable fall damage so the bot survives the 180-block plunge from
  // Y=250. Idempotent; harmless if a peer bot already set it.
  bot.chat('/gamerule fallDamage false');
  // Y=250 above any terrain; bot falls to surface (needs
  // allow-flight=true to tolerate the brief airborne phase).
  bot.chat(`/tp ${tx} 250 ${tz}`);
  // Wait for the forced teleport, then poll until the bot is actually
  // on the ground before opening the bridge — pathfinder commanding a
  // mid-air bot produces "Invalid move player packet" kicks.
  bot.once('forcedMove', () => {
    const deadline = Date.now() + LAND_TIMEOUT_MS;
    const waitLanded = () => {
      if (bot.entity && bot.entity.onGround) {
        const p = bot.entity.position;
        console.log(`bot ${ID} landed at (${p.x.toFixed(0)}, ${p.y.toFixed(0)}, ${p.z.toFixed(0)})`);
        startServer();
      } else if (Date.now() >= deadline) {
        console.log(`bot ${ID} land timeout; opening bridge anyway`);
        startServer();
      } else {
        setTimeout(waitLanded, 250);
      }
    };
    setTimeout(waitLanded, 500);  // grace period after forcedMove
  });
});

bot.on('error', (e) => console.error(`bot ${ID} error:`, e.message));
bot.on('death', () => console.log(`bot ${ID} died`));

// Sample a (2r+1)x(2r+1) biome grid on a 4-block stride, centered on
// the bot's current cell. Y is fixed to the bot's current Y — we only
// ever locomote on the surface, so there's no need to scan vertically.
// Returns { cellX, cellZ, grid }.
function sampleGrid(r) {
  const p = bot.entity.position;
  const y = Math.floor(p.y);
  const cellX = Math.floor(p.x / 4);
  const cellZ = Math.floor(p.z / 4);
  const size = 2 * r + 1;
  const grid = new Array(size * size);
  for (let dz = -r; dz <= r; dz++) {
    for (let dx = -r; dx <= r; dx++) {
      const bx = (cellX + dx) * 4 + 2;  // sample at cell center
      const bz = (cellZ + dz) * 4 + 2;
      if (!visible(bx, bz)) {
        grid[(dz + r) * size + (dx + r)] = -1;
        continue;
      }
      const b = bot.world.getBiome({ x: bx, y, z: bz });
      // Overworld generation never produces the_void (id 0); a 0 read
      // means the chunk hasn't streamed in yet. Surface that as "unknown"
      // through the same -1 channel reserved for line-of-sight.
      grid[(dz + r) * size + (dx + r)] = b === 0 ? -1 : b;
    }
  }
  return { cellX, cellZ, grid };
}

function getObs() {
  const p = bot.entity.position;
  const cellX = Math.floor(p.x / 4);
  const cellZ = Math.floor(p.z / 4);
  const biomeId = bot.world.getBiome({ x: Math.floor(p.x), y: Math.floor(p.y), z: Math.floor(p.z) });
  bot.visitedBiomes.add(biomeId);
  const obs = {
    biomeId,
    biomeName: mcData.biomes[biomeId]?.name ?? 'unknown',
    cellX,
    cellZ,
    x: Math.floor(p.x),
    z: Math.floor(p.z),
    health: bot.health,
    food: bot.food,
    numVisited: bot.visitedBiomes.size,
    visitedBiomes: [...bot.visitedBiomes],
  };
  if (WORLD_MODE === 'los') {
    const sample = sampleGrid(GRID_RADIUS);
    obs.gridRadius = GRID_RADIUS;
    obs.grid = sample.grid;
  }
  return obs;
}

const ACTION_TIMEOUT_MS = 30000;
const BIOME_SAMPLE_MS = 1000;  // 20 ticks; biome cells are 4 blocks wide

// When an action returns STUCK, dump the immediate terrain around the
// bot so we can grep / aggregate the actual reasons mineflayer-pathfinder
// is giving up. Logs as a single grep-able line:
//   [stuck-detail bot=N] theta=θ feet=X feet_below=Y head=Z front_feet=A
//     front_head=B front_above=C front_below=D onGround=bool inWater=bool
//     health=H/20 food=F/20 inv=hand_block (count)
// Where "front_*" probes one cell in the requested compass direction at
// foot / head / above-head / below-foot y-levels — enough to identify
// the canonical stuck causes (wall, ledge, leaves, water, lava).
function logStuckTerrain(theta, sx, sy, sz) {
  const rad = (theta * Math.PI) / 180;
  const dx = Math.round(Math.sin(rad));
  const dz = Math.round(Math.cos(rad));
  const fx = sx + dx, fz = sz + dz;
  const nameAt = (x, y, z) => {
    const b = bot.blockAt(new Vec3(x, y, z));
    return b ? b.name : 'unloaded';
  };
  const p = bot.entity && bot.entity.position;
  const inWater = !!(bot.entity && (bot.entity.isInWater || bot.entity.isInLava));
  const onGround = bot.entity ? !!bot.entity.onGround : false;
  const heldItem = bot.heldItem ? `${bot.heldItem.name}(${bot.heldItem.count})` : 'none';
  console.log(
    `[stuck-detail bot=${ID}] theta=${theta.toFixed(0)} ` +
    `feet=${nameAt(sx, sy, sz)} ` +
    `feet_below=${nameAt(sx, sy - 1, sz)} ` +
    `head=${nameAt(sx, sy + 1, sz)} ` +
    `front_feet=${nameAt(fx, sy, fz)} ` +
    `front_head=${nameAt(fx, sy + 1, fz)} ` +
    `front_above=${nameAt(fx, sy + 2, fz)} ` +
    `front_below=${nameAt(fx, sy - 1, fz)} ` +
    `onGround=${onGround} inWater=${inWater} ` +
    `health=${(bot.health ?? 0).toFixed(0)}/20 food=${bot.food ?? 0}/20 ` +
    `hand=${heldItem}`);
}


function executeAction({ theta, distance }, cb) {
  if (!distance) { cb(getObs()); return; }
  const rad = (theta * Math.PI) / 180;
  const p = bot.entity.position;
  const sx = Math.floor(p.x), sy = Math.floor(p.y), sz = Math.floor(p.z);
  const tx = p.x + Math.sin(rad) * distance;
  const tz = p.z + Math.cos(rad) * distance;
  const startBiomeId = bot.world.getBiome({ x: sx, y: sy, z: sz });
  const startBiome = mcData.biomes[startBiomeId]?.name ?? 'unknown';
  const t0 = Date.now();
  const goal = new GoalNearXZ(tx, tz, 16);
  bot.pathfinder.setGoal(goal, false);

  // Sample biome every ~1s during the hop so we catch mid-traversal
  // biomes that the start/end snapshot would miss. Pathfinder walks
  // ~4.3 b/s and biome cells are 4 blocks wide -> ~1 sample per cell.
  let midSamples = 0;
  const sampler = setInterval(() => {
    if (!bot.entity || !bot.entity.position) return;
    const q = bot.entity.position;
    const bid = bot.world.getBiome(
      { x: Math.floor(q.x), y: Math.floor(q.y), z: Math.floor(q.z) });
    bot.visitedBiomes.add(bid);
    midSamples++;
  }, BIOME_SAMPLE_MS);

  let done = false;
  const finish = (stuck, reason) => {
    if (done) return;
    done = true;
    clearTimeout(timer);
    clearInterval(sampler);
    bot.removeListener('goal_reached', onReach);
    bot.removeListener('path_update', onStuck);
    bot.pathfinder.setGoal(null);
    const obs = stuck ? { ...getObs(), stuck: true } : getObs();
    const moved = Math.hypot(obs.x - sx, obs.z - sz);
    const dt = ((Date.now() - t0) / 1000).toFixed(1);
    // Single-line, grep-able per-move trace in block coordinates.
    console.log(
      `[move bot=${ID}] theta=${theta.toFixed(0)} d=${distance} ` +
      `start=(${sx},${sy},${sz}) startBiome=${startBiome} ` +
      `target=(${Math.round(tx)},${Math.round(tz)}) ` +
      `end=(${obs.x},${obs.z}) endBiome=${obs.biomeName} ` +
      `moved=${moved.toFixed(1)} dt=${dt}s samples=${midSamples} ` +
      `result=${stuck ? `STUCK:${reason}` : 'OK'}`);
    if (stuck) logStuckTerrain(theta, sx, sy, sz);
    cb(obs);
  };
  const onReach = () => finish(false, 'reached');
  const onStuck = (r) => {
    if (r.status === 'noPath' || r.status === 'timeout') finish(true, r.status);
  };
  const timer = setTimeout(() => finish(true, 'action-timeout'), ACTION_TIMEOUT_MS);
  bot.once('goal_reached', onReach);
  bot.on('path_update', onStuck);
}

function startServer() {
  const server = net.createServer((conn) => {
    console.log(`bot ${ID} agent connected`);
    let buf = '';
    conn.on('data', (chunk) => {
      buf += chunk.toString();
      let idx;
      while ((idx = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, idx);
        buf = buf.slice(idx + 1);
        if (!line.trim()) continue;
        try {
          const action = JSON.parse(line);
          executeAction(action, (obs) => conn.write(JSON.stringify(obs) + '\n'));
        } catch (e) {
          console.error(`bot ${ID} parse error:`, e.message);
        }
      }
    });
  });
  server.listen(BRIDGE_PORT, () => console.log(`bot ${ID} listening on ${BRIDGE_PORT}`));
}
