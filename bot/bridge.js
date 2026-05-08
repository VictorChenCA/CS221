// JSON-over-TCP bridge between the Python agent and a mineflayer bot.
// Protocol: one JSON object per line. Requests {cmd, args}, replies {ok, data}.

const net = require('net');
const mineflayer = require('mineflayer');
const { pathfinder, Movements, goals } = require('mineflayer-pathfinder');

const HOST = process.env.MC_HOST || 'localhost';
const PORT = parseInt(process.env.MC_PORT || '25565', 10);
const BRIDGE_PORT = parseInt(process.env.BRIDGE_PORT || '8765', 10);

function makeBot(username) {
  const bot = mineflayer.createBot({ host: HOST, port: PORT, username });
  bot.loadPlugin(pathfinder);
  return bot;
}

// TODO: dispatch table for cmds: observe, move(theta, d), reset
// TODO: handle pathfinder timeouts, fall in water, etc.

const server = net.createServer((sock) => {
  // one bot per connection for now
  const bot = makeBot('agent_' + Date.now());
  // ...
});

server.listen(BRIDGE_PORT, () => {
  console.log(`bridge listening on ${BRIDGE_PORT}`);
});
