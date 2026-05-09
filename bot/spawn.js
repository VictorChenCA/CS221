// Launch N bridge.js processes, staggered by 2s to avoid server overload.
// Each child gets its own bot id (=> port 9000+id).
//   usage: node bot/spawn.js [N=10]

const { spawn } = require('child_process');
const path = require('path');

const N = parseInt(process.argv[2] || '10', 10);
const STAGGER_MS = 2000;

for (let i = 0; i < N; i++) {
  setTimeout(() => {
    const child = spawn('node', [path.join(__dirname, 'bridge.js'), String(i)], {
      stdio: 'inherit',
    });
    child.on('exit', (code) => console.log(`bot ${i} exited (${code})`));
  }, i * STAGGER_MS);
}
