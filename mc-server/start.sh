#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export JAVA_HOME="$(/usr/libexec/java_home -v 21)"
exec "$JAVA_HOME/bin/java" -Xmx6G -Xms2G -jar paper.jar nogui
