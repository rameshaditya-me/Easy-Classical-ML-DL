#!/bin/sh
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
rm -rf "$ROOT/docs/pages"
cp -R "$ROOT/src/pages" "$ROOT/docs/pages"
echo "Synced src/pages → docs/pages"
