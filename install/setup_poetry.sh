#!/usr/bin/env bash
set -e

echo "📦 Setting up Poetry..."

if ! command -v poetry >/dev/null; then
  echo "⬇️ Installing Poetry"
  curl -sSL https://install.python-poetry.org | python3 -
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "🐍 Creating virtual environment"
poetry env use python3
poetry install --no-interaction --no-ansi
