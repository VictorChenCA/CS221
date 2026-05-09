// One mineflayer bot per process. Listens on TCP port 9000+id, speaks
// newline-delimited JSON to the Python agent.
//   request:  {"theta": 90, "distance": 100}
//   response: {"biomeId", "biomeName", "x", "z", "health", "food",
//              "numVisited", "visitedBiomes", "stuck"}

const net = require('net');
const mineflayer = require('mineflayer');
const { pathfinder, Movements, goals: { GoalXZ } } = require('mineflayer-pathfinder');
const mcData = require('minecraft-data')('1.20.1');

const ID = parseInt(process.argv[2] || '0', 10);
const HOST = process.env.MC_HOST || 'localhost';
const PORT = parseInt(process.env.MC_PORT || '25565', 10);
const BRIDGE_PORT = 9000 + ID;

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

function getObs() {
  const p = bot.entity.position;
  const biomeId = bot.world.getBiome(Math.floor(p.x), Math.floor(p.y), Math.floor(p.z));
  bot.visitedBiomes.add(biomeId);
  return {
    biomeId,
    biomeName: mcData.biomes[biomeId]?.name ?? 'unknown',
    x: Math.floor(p.x),
    z: Math.floor(p.z),
    health: bot.health,
    food: bot.food,
    numVisited: bot.visitedBiomes.size,
    visitedBiomes: [...bot.visitedBiomes],
  };
}

function executeAction({ theta, distance }, cb) {
  const rad = (theta * Math.PI) / 180;
  const p = bot.entity.position;
  const goal = new GoalXZ(p.x + Math.sin(rad) * distance, p.z + Math.cos(rad) * distance);
  bot.pathfinder.setGoal(goal, true);

  const onReach = () => {
    bot.removeListener('path_update', onStuck);
    cb(getObs());
  };
  const onStuck = (r) => {
    if (r.status === 'noPath' || r.status === 'timeout') {
      bot.removeListener('goal_reached', onReach);
      cb({ ...getObs(), stuck: true });
    }
  };
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
