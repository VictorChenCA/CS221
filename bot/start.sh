#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
n="${1:-10}"
exec node bot/spawn.js "$n"
