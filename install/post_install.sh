#!/usr/bin/env bash
set -e

echo "🧪 Running sanity checks..."

poetry run python - <<EOF
import sys
print("omid_llm OK on Python", sys.version)
EOF
