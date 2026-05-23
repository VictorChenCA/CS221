// Diagnose why bot.world.getBiome returns 0 ("badlands") for everything.
// Connects to localhost:25565 as a single bot, waits for chunks, then
// dumps the chunk's biome storage so we can see whether (a) the chunk's
// biome section is undefined (decoder bug or chunks-not-loaded) or
// (b) it's populated but holds the wrong data.

const mineflayer = require('mineflayer');
const bot = mineflayer.createBot({
  host: 'localhost', port: 25565,
  username: 'Diag', auth: 'offline', version: '1.20.1',
});

bot.on('error', (e) => console.log('ERR:', e.message));

bot.once('spawn', async () => {
  console.log('spawned, waiting 4s for chunks');
  await new Promise(r => setTimeout(r, 4000));

  const p = bot.entity.position;
  console.log(`position = (${p.x.toFixed(1)}, ${p.y.toFixed(1)}, ${p.z.toFixed(1)})`);

  const cx = Math.floor(p.x / 16), cz = Math.floor(p.z / 16);
  console.log(`chunk = (${cx}, ${cz})`);

  const col = bot.world.getColumn ? bot.world.getColumn(cx, cz) : null;
  if (!col) {
    console.log('NO COLUMN — chunk not loaded');
    bot.quit(); process.exit(0);
  }
  console.log(`column class: ${col.constructor.name}, minY=${col.minY}, biomes array length=${col.biomes?.length}`);
  if (!col.biomes) {
    console.log('col.biomes is missing entirely');
    bot.quit(); process.exit(0);
  }

  // Inspect every section's biome storage
  for (let i = 0; i < col.biomes.length; i++) {
    const b = col.biomes[i];
    const yBase = col.minY + i * 16;
    if (!b) {
      console.log(`  section ${i} (yBase=${yBase}): UNDEFINED`);
      continue;
    }
    let palette = '?';
    if (b.palette) palette = JSON.stringify(b.palette);
    else if (b.value !== undefined) palette = `single=${b.value}`;
    console.log(`  section ${i} (yBase=${yBase}): class=${b.constructor.name} palette=${palette} bitsPerEntry=${b.bitsPerValue ?? b.bitsPerEntry ?? '?'}`);
  }

  // Sample biome at a few positions
  console.log('\nGetBiome samples around bot:');
  for (const [dx, dy, dz] of [[0,0,0],[10,0,0],[0,4,0],[0,-4,0],[16,0,16],[100,0,100]]) {
    const bid = bot.world.getBiome({ x: Math.floor(p.x)+dx, y: Math.floor(p.y)+dy, z: Math.floor(p.z)+dz });
    console.log(`  (+${dx},+${dy},+${dz}) -> id=${bid}`);
  }

  bot.quit();
  setTimeout(() => process.exit(0), 500);
});
