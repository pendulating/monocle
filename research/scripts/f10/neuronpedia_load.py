#!/usr/bin/env python
"""Put the F10.6 models and runs into the local Neuronpedia database.

The viewer fetches a run with `force-cache`, thus a browser keeps the old file at the
same URL. Each load adds a new stamp to the URL, and the browser fetches the new file.

The script writes to the private Postgres of `neuronpedia_up.sh` only. A second call
replaces the rows, thus you can call it again after a new export.

Use (the Python of `neuronpedia_up.sh` holds `psycopg2`):
    python research/scripts/f10/neuronpedia_load.py data/f10/hop_all/viewer
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

DSN = "host=127.0.0.1 port=55432 user=postgres password=postgres dbname=postgres"
CREATOR = "clkht01d40000jv08hvalcvly"
MODELS = {
    "gemma-4-31b": ("Gemma-4-31B", 59, 5376, "Google"),
    "qwen3.5-9b": ("Qwen3.5-9B", 31, 4096, "Qwen"),
    "qwen3.6-27b": ("Qwen3.6-27B", 63, 5120, "Qwen"),
}
COLUMNS = ("id", "kind", "modelId", "url", "description", "selectedPositions", "lensColumns", "activeLensModeTab", "topN",
           "hideNonWordTokens", "temperature", "numCompletionTokens", "numPromptTokens")


def main() -> None:
    root = Path(sys.argv[1])
    stamp = int(time.time())
    with psycopg2.connect(DSN) as db, db.cursor() as c:
        for model, (name, layers, width, owner) in MODELS.items():
            if not (root / model / "shares.json").exists():
                continue
            c.execute(
                'INSERT INTO "Model" (id, "displayNameShort", "displayName", "creatorId", layers, "neuronsPerLayer", owner, visibility, instruct, "updatedAt") '
                "VALUES (%s, %s, %s, %s, %s, %s, %s, 'PUBLIC', true, now()) ON CONFLICT (id) DO NOTHING",
                (model, name, name, CREATOR, layers, width, owner))
            rows = json.loads((root / model / "shares.json").read_text())
            c.execute('DELETE FROM "JlensShare" WHERE "modelId" = %s', (model,))
            names = ", ".join(f'"{k}"' for k in COLUMNS)
            execute_values(c, f'INSERT INTO "JlensShare" ({names}, "updatedAt") VALUES %s',
                           [tuple({**r, "url": f"{r['url']}?v={stamp}"}[k] for k in COLUMNS) for r in rows], template="(" + ", ".join(["%s"] * len(COLUMNS)) + ", now())")
            print(f"{model}: {len(rows)} runs")


if __name__ == "__main__":
    main()
