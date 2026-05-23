// Three-way biome accuracy check at probe coords:
//   cubiomes (offline, what our pipeline uses)
//   mineflayer's bot.world.getBiome (the field we've been overriding)
//   Paper's authoritative answer via /locate biome <name>
//
// Method for Paper's answer: we ask the server to locate the nearest
// instance of each candidate biome. The biome found at the smallest
// distance (= 0 if we're standing in it) is Paper's verdict at our
// coord. We test against the cubiomes-named biomes our seed actually
// contains, so the search has bounded cardinality.
//
//   usage: node tools/biome_check.js <seed> <mc_port>

const mineflayer = require('mineflayer');
const mcData = require('minecraft-data')('1.20.1');
const { execSync } = require('child_process');
const fs = require('fs');

const SEED = parseInt(process.argv[2] || '123', 10);
const PORT = parseInt(process.argv[3] || '25565', 10);
const PROBES = [
  [0, 0], [250, 0], [0, 250], [-250, 0], [0, -250],
  [250, 250], [-250, -250], [500, 500], [-500, 500],
];

// Cubiomes name table (built from biomes.h earlier; cached in /tmp).
const cubeNames = JSON.parse(fs.readFileSync('/tmp/cube_names.json', 'utf8'));

const bot = mineflayer.createBot({
  host: 'localhost', port: PORT, username: 'Raz0rMC',
  auth: 'offline', version: '1.20.1',
});

bot.on('error', (e) => { console.error('error:', e.message); process.exit(1); });

bot.once('spawn', async () => {
  await sleep(2000);

  // 1) Bulk cubiomes lookup via Python subprocess.
  const py = [
    "import sys; sys.path.insert(0, '.')",
    "from mdp.biomegen import cubiomes_gen",
    `g = cubiomes_gen(${SEED})`,
    `probes = ${JSON.stringify(PROBES)}`,
    "for x, z in probes:",
    "    print(f'{x} {z} {g(x // 4, z // 4)}')",
  ].join('\n');
  fs.writeFileSync('/tmp/cube_probe.py', py);
  const cubeOut = execSync('python3 /tmp/cube_probe.py', { cwd: process.cwd() }).toString();
  const cubeById = new Map();
  for (const line of cubeOut.trim().split('\n')) {
    const [x, z, id] = line.split(' ').map(Number);
    cubeById.set(`${x},${z}`, id);
  }

  // 2) For Paper's verdict, /tp the bot to each probe, then read the
  //    server's chat tab-completion list for /locate biome (which is
  //    too noisy) — instead, send `/data get entity @s` for the
  //    `Brain.memories.minecraft:biome` field. That field doesn't
  //    exist in 1.20.1. Fall back to /locate biome <name> with
  //    distance: each cubiomes id we see at any probe → try locating
  //    it; nearest distance tells us if we're standing in it.
  //
  // Simpler authoritative read: use `/data get entity @s` won't give
  // biome. So we instead send `/tp ~ ~ ~` (no-op), then use the
  // server's reply to `/locate biome minecraft:<name>` and observe
  // distance — distance 0 means "at this position".
  //
  // For each probe, we ask /locate for the cubiomes-named biome at
  // that probe; if Paper agrees, distance from probe will be small.

  // Collect server replies — /locate output goes via systemChat in 1.20.1.
  const replies = [];
  bot.on('message', (msg) => { replies.push(msg.toString()); });
  bot._client.on('system_chat', (pkt) => {
    try {
      const j = JSON.parse(pkt.content);
      const text = (j.text || '') + (j.with ? JSON.stringify(j.with) : '') + (j.extra ? JSON.stringify(j.extra) : '');
      replies.push(text);
    } catch { replies.push(String(pkt.content)); }
  });

  console.log(`${'x'.padStart(5)} ${'z'.padStart(5)} | ${'mf id'.padStart(6)} ${'mf name'.padStart(14)} | ${'cube id'.padStart(7)} ${'cube name'.padStart(20)} | paper /locate near?`);
  console.log('-'.repeat(100));

  for (const [x, z] of PROBES) {
    // /tp self to probe
    bot.chat(`/tp @s ${x} 100 ${z}`);
    await sleep(1500);  // teleport + chunk load
    const y = Math.floor(bot.entity.position.y);
    const mfId = bot.world.getBiome(x, y, z);
    const mfName = mcData.biomes[mfId]?.name ?? '?';
    const cubeId = cubeById.get(`${x},${z}`);
    const cubeName = cubeNames[String(cubeId)] ?? '?';

    // Ask Paper to find nearest of cube's claimed biome.
    replies.length = 0;
    bot.chat(`/locate biome minecraft:${cubeName}`);
    await sleep(1000);
    // Parse a reply like "The nearest minecraft:plains is at [123, ~, 456] (5 blocks away)"
    const r = replies.find(s => /minecraft:|nearest|away|could not be found/.test(s)) ?? '(no reply)';
    const dist = (r.match(/\((\d+) blocks? away\)/) || [])[1];
    const summary = dist !== undefined ? `dist=${dist}` : (r.includes('could not') ? 'NOT FOUND' : r.slice(0, 60));

    console.log(`${String(x).padStart(5)} ${String(z).padStart(5)} | ${String(mfId).padStart(6)} ${mfName.padStart(14)} | ${String(cubeId).padStart(7)} ${cubeName.padStart(20)} | ${summary}`);
  }
  bot.quit();
  process.exit(0);
});

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
