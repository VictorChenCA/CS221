// Spawn N bots evenly on a circle of radius R around (0,0).
// Used to parallelize trajectory collection on a single world.

const N = parseInt(process.argv[2] || '10', 10);
const R = parseInt(process.argv[3] || '500', 10);

for (let i = 0; i < N; i++) {
  const theta = (2 * Math.PI * i) / N;
  const x = Math.round(R * Math.cos(theta));
  const z = Math.round(R * Math.sin(theta));
  console.log(`bot ${i}: spawn at (${x}, ${z})`);
  // TODO: launch bridge worker, teleport on join
}
