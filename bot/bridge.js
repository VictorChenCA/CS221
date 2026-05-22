// One mineflayer bot per process. Listens on TCP port 9000+id, speaks
// newline-delimited JSON to the Python agent.
//   request:  {"theta": 90, "distance": 100}
//   response: {"biomeId", "biomeName", "cellX", "cellZ", "x", "z",
//              "health", "food", "numVisited", "visitedBiomes",
//              "gridRadius", "grid", "stuck"}
//
// The world is sampled on a 4-block stride to match Minecraft 1.18+'s
// native biome cell. `grid` is a flat (2r+1)x(2r+1) row-major array of
// biome ids in row-major order with +dz as the outer index. A value of
// -1 means "not visible" (currently unused; reserved for line-of-sight).

const net = require('net');
const mineflayer = require('mineflayer');
const { pathfinder, Movements, goals: { GoalNearXZ } } = require('mineflayer-pathfinder');
const mcData = require('minecraft-data')('1.20.1');

const ID = parseInt(process.argv[2] || '0', 10);
const HOST = process.env.MC_HOST || 'localhost';
const PORT = parseInt(process.env.MC_PORT || '25565', 10);
const BRIDGE_PORT = 9000 + ID;
const GRID_RADIUS = parseInt(process.env.GRID_RADIUS || '32', 10);  // cells

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
  console.log(`bot ${ID} spawned`);
  startServer();
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
      grid[(dz + r) * size + (dx + r)] = visible(bx, bz)
        ? bot.world.getBiome(bx, y, bz)
        : -1;
    }
  }
  return { cellX, cellZ, grid };
}

function getObs() {
  const { cellX, cellZ, grid } = sampleGrid(GRID_RADIUS);
  const centerIdx = (GRID_RADIUS) * (2 * GRID_RADIUS + 1) + GRID_RADIUS;
  const biomeId = grid[centerIdx];
  bot.visitedBiomes.add(biomeId);
  const p = bot.entity.position;
  return {
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
    gridRadius: GRID_RADIUS,
    grid,
  };
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
