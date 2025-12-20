#!/usr/bin/env bash
set -e

REPO_URL="https://github.com/omaghsoudi/LLM_RAGs"
INSTALL_DIR="$HOME/.omid_llm"

echo "🚀 Installing omid_llm..."

# Clone or update
if [ -d "$INSTALL_DIR" ]; then
  echo "🔄 Updating existing install"
  cd "$INSTALL_DIR"
  git pull
else
  echo "📦 Cloning repository"
  git clone "$REPO_URL" "$INSTALL_DIR"
  cd "$INSTALL_DIR"
fi

bash install/check_prereqs.sh
bash install/setup_poetry.sh
bash install/post_install.sh

echo "✅ omid_llm installed successfully"
echo "👉 Run: cd $INSTALL_DIR"
echo "👉 Then: poetry run omid-llm"


echo "✅ Done"