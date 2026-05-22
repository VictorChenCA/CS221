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

const ID = parseInt(process.argv[2] || '0', 10);
const HOST = process.env.MC_HOST || 'localhost';
const PORT = parseInt(process.env.MC_PORT || '25565', 10);
const BRIDGE_PORT = 9000 + ID;
const GRID_RADIUS = parseInt(process.env.GRID_RADIUS || '8', 10);  // cells (1 cell = 4 blocks)
const SPAWN_CHUNK_WAIT_MS = 2000;  // let initial chunk stream settle before serving the first obs
// Proposal §6: spawn bots on a circle so they don't pathfinder-collide
// at (0,0). DISPERSE_R = circle radius in blocks; DISPERSE_N = how many
// bots are spread around this server (used for the angle).
const DISPERSE_R = parseInt(process.env.DISPERSE_R || '500', 10);
const DISPERSE_N = parseInt(process.env.DISPERSE_N || '5', 10);
const DISPERSE_WAIT_MS = 3000;  // give the /tp packet + new chunks time to settle
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
  bot.pathfinder.setMovements(new Movements(bot));
  // Disperse on a DISPERSE_R-block circle (proposal §6). The /tp
  // command needs op — see tools/run_test_eval.py which writes ops.json.
  const localId = ID % DISPERSE_N;
  const angle = (localId / DISPERSE_N) * 2 * Math.PI;
  const tx = Math.round(Math.cos(angle) * DISPERSE_R);
  const tz = Math.round(Math.sin(angle) * DISPERSE_R);
  console.log(`bot ${ID} spawned, dispersing to (${tx}, ${tz})`);
  bot.chat(`/tp ${tx} 100 ${tz}`);
  setTimeout(() => {
    console.log(`bot ${ID} ready at`,
      bot.entity && bot.entity.position
        ? `(${bot.entity.position.x.toFixed(0)}, ${bot.entity.position.z.toFixed(0)})`
        : '(no position?)');
    startServer();
  }, SPAWN_CHUNK_WAIT_MS + DISPERSE_WAIT_MS);
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
      const b = bot.world.getBiome(bx, y, bz);
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
  const biomeId = bot.world.getBiome(Math.floor(p.x), Math.floor(p.y), Math.floor(p.z));
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

function executeAction({ theta, distance }, cb) {
  if (!distance) { cb(getObs()); return; }
  const rad = (theta * Math.PI) / 180;
  const p = bot.entity.position;
  const tx = p.x + Math.sin(rad) * distance;
  const tz = p.z + Math.cos(rad) * distance;
  const goal = new GoalNearXZ(tx, tz, 3);
  bot.pathfinder.setGoal(goal, false);

  let done = false;
  const finish = (stuck) => {
    if (done) return;
    done = true;
    clearTimeout(timer);
    bot.removeListener('goal_reached', onReach);
    bot.removeListener('path_update', onStuck);
    bot.pathfinder.setGoal(null);
    cb(stuck ? { ...getObs(), stuck: true } : getObs());
  };
  const onReach = () => finish(false);
  const onStuck = (r) => {
    if (r.status === 'noPath' || r.status === 'timeout') finish(true);
  };
  const timer = setTimeout(() => finish(true), ACTION_TIMEOUT_MS);
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
