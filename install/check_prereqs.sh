#!/usr/bin/env bash
set -e

echo "🔍 Checking prerequisites..."

command -v git >/dev/null || { echo "❌ git not found"; exit 1; }
command -v curl >/dev/null || { echo "❌ curl not found"; exit 1; }
command -v python3 >/dev/null || { echo "❌ python3 not found"; exit 1; }

python3 - <<EOF
import sys
assert sys.version_info >= (3,9), "Python 3.9+ required"
print("✅ Python OK:", sys.version)
EOF
