// One-shot diagnostic: connect to a running test server, read raw biome
// IDs at several coords, look them up in minecraft-data, compare to
// cubiomes_gen output for the same coords. Prints a small table.
//
//   usage: node tools/biome_check.js <seed> <mc_port>

const mineflayer = require('mineflayer');
const mcData = require('minecraft-data')('1.20.1');
const { execSync } = require('child_process');

const SEED = parseInt(process.argv[2] || '123', 10);
const PORT = parseInt(process.argv[3] || '25565', 10);

const probes = [
  [0, 0], [250, 0], [0, 250], [-250, 0], [0, -250],
  [250, 250], [-250, -250], [500, 0], [0, 500],
];

const bot = mineflayer.createBot({
  host: 'localhost', port: PORT, username: 'BiomeProbe',
  auth: 'offline', version: '1.20.1',
});

bot.on('error', (e) => { console.error('error:', e.message); process.exit(1); });

bot.once('spawn', async () => {
  await new Promise(r => setTimeout(r, 2000));  // chunks settle

  // One cubiomes call gives biome at every probe; do it in Python.
  const fs = require('fs');
  const py = [
    "import sys; sys.path.insert(0, '.')",
    "from mdp.biomegen import cubiomes_gen",
    `g = cubiomes_gen(${SEED})`,
    `probes = ${JSON.stringify(probes)}`,
    "for x, z in probes:",
    "    cx, cz = x // 4, z // 4",
    "    print(f'{x} {z} {g(cx, cz)}')",
  ].join('\n');
  fs.writeFileSync('/tmp/cube_probe.py', py);
  const out = execSync('python3 /tmp/cube_probe.py', { cwd: process.cwd() }).toString();
  const cubeAt = new Map();
  for (const line of out.trim().split('\n')) {
    const [x, z, id] = line.split(' ').map(Number);
    cubeAt.set(`${x},${z}`, id);
  }

  console.log(`${'x'.padStart(5)} ${'z'.padStart(5)} | ${'mineflayer id'.padStart(13)} ${'mc-data name'.padStart(28)} | ${'cubiomes id'.padStart(11)}`);
  console.log('-'.repeat(80));
  for (const [x, z] of probes) {
    const y = Math.floor(bot.entity.position.y);
    const mfId = bot.world.getBiome(x, y, z);
    const name = mcData.biomes[mfId]?.name ?? '?';
    const cubeId = cubeAt.get(`${x},${z}`);
    console.log(`${String(x).padStart(5)} ${String(z).padStart(5)} | ${String(mfId).padStart(13)} ${name.padStart(28)} | ${String(cubeId).padStart(11)}`);
  }
  bot.quit();
  process.exit(0);
});
