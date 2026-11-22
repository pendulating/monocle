#!/usr/bin/env bash
# Start the local Neuronpedia viewer for the F10.6 runs on this node.
# A second call reuses the install, replaces the runs and restarts the server.
# The app and the database listen on 127.0.0.1 only. A gate gives the login node the app port.
# See vlm-narratives-docs/neuronpedia-viewer.md.
#
#   bash research/scripts/f10/neuronpedia_up.sh
set -euo pipefail
ROOT=/share/ju/matt/mllmsci
BASE=/scratch/$USER/neuronpedia
PIN=e2573d3
RUNS=${F10_VIEWER_RUNS:-$ROOT/data/f10/hop_all/viewer}
PORT=${F10_VIEWER_PORT:-3000}
INNER=$((PORT + 1))
# The address of unicorn-login-04. The gate refuses all other addresses, except this node's 127.0.0.1.
ALLOW=${F10_VIEWER_ALLOW:-128.84.96.23}
WEB=$BASE/src/apps/webapp
mkdir -p "$BASE"

[ -d "$BASE/src" ] || git clone --filter=blob:none -q https://github.com/hijohnnylin/neuronpedia.git "$BASE/src"
git -C "$BASE/src" checkout -q "$PIN" 2>/dev/null || { git -C "$BASE/src" checkout -q -- . && git -C "$BASE/src" checkout -q "$PIN"; }

if [ ! -x "$BASE/pgvenv/bin/python" ]; then
  uv venv -q --python 3.12 "$BASE/pgvenv"
  uv pip install -q --python "$BASE/pgvenv/bin/python" pgserver psycopg2-binary
fi
PG=$("$BASE/pgvenv/bin/python" -c "import pgserver,pathlib;print(pathlib.Path(pgserver.__file__).parent/'pginstall'/'bin')")
export PATH=$PG:$PATH NEXT_TELEMETRY_DISABLED=1

[ -d "$WEB/node_modules" ] || (cd "$WEB" && npm ci --no-audit --no-fund)

# Local patch: the second column holds the contrast, not a logit lens.
sed -i "s/\[LensType.LOGIT_LENS\]: 'Logit Lens'/[LensType.LOGIT_LENS]: 'Contrast'/" "$WEB/lib/utils/lens.ts"
sed -i "s/\[LensColumn.LOGIT_LENS\]: 'Logit'/[LensColumn.LOGIT_LENS]: 'Contrast'/" "$WEB/lib/utils/lens.ts"
sed -i 's/localhost:5432/localhost:55432/g' "$WEB/.env.localhost"
sed -i 's/^NEXT_PUBLIC_JPP_LENS_MODEL_IDS=.*/NEXT_PUBLIC_JPP_LENS_MODEL_IDS=gemma-4-31b,qwen3.5-9b,qwen3.6-27b/' "$WEB/.env.localhost"

FIRST=0
if [ ! -d "$BASE/pgdata" ]; then
  FIRST=1
  printf postgres > "$BASE/pw.txt"
  initdb -D "$BASE/pgdata" -U postgres --pwfile="$BASE/pw.txt" -A scram-sha-256 -E UTF8 > "$BASE/initdb.log"
  rm "$BASE/pw.txt"
fi
pg_ctl -D "$BASE/pgdata" status > /dev/null 2>&1 ||
  pg_ctl -D "$BASE/pgdata" -o "-p 55432 -c listen_addresses=127.0.0.1 -k $BASE" -l "$BASE/pg.log" -w start > /dev/null
if [ "$FIRST" = 1 ]; then
  (cd "$WEB"
   npx --no-install env-cmd -f .env.localhost --use-shell 'prisma generate'
   npx --no-install env-cmd -f .env.localhost --use-shell 'prisma migrate deploy'
   npx --no-install env-cmd -f .env.localhost --use-shell 'prisma db seed'
   npx --no-install env-cmd -f .env.localhost --use-shell 'psql "$POSTGRES_URL_NON_POOLING" -q -f prisma/pgvector-init/pgvector.sql') 2>&1 | grep -v 'npm notice' | tail -4
fi

mkdir -p "$WEB/public/f10"
rsync -a --delete "$RUNS/" "$WEB/public/f10/"
"$BASE/pgvenv/bin/python" "$ROOT/research/scripts/f10/neuronpedia_load.py" "$RUNS"

pkill -u "$USER" -f "neuronpedia_gate.py --port $PORT" 2>/dev/null || true
pkill -u "$USER" -f "next dev --turbopack -H 127.0.0.1" 2>/dev/null || true
pkill -u "$USER" -f "next-server" 2>/dev/null || true
sleep 1
cd "$WEB"
nohup npx --no-install env-cmd -f .env.localhost --use-shell \
  "npx --no-install next dev --turbopack -H 127.0.0.1 -p $INNER" > "$BASE/web.log" 2>&1 &
nohup "$BASE/pgvenv/bin/python" "$ROOT/research/scripts/f10/neuronpedia_gate.py" --port "$PORT" --target "$INNER" --allow $ALLOW > "$BASE/gate.log" 2>&1 &
echo "viewer: http://localhost:$PORT/f10/index.html through the forward to $(hostname):$PORT, logs $BASE/web.log, $BASE/gate.log"
