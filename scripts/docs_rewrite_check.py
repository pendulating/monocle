"""Check that a rewrite of a doc kept its data.

Compares a rewritten markdown file with its backup. The rewrite may
change words. It must not change a number, a code span, a file path or a
table shape. The script prints each item that the backup holds and the
rewrite does not (and the reverse), as a multiset difference.

    .venv-vllm025cu129/bin/python scripts/docs_rewrite_check.py <backup.md> <new.md>
"""
from __future__ import annotations

import re
import sys
from collections import Counter

NUM = re.compile(r"(?<![A-Za-z_])[-+−]?\d[\d,]*\.?\d*(?:e[-+]?\d+)?%?")
CODE = re.compile(r"`([^`\n]+)`")


def items(text: str) -> dict[str, Counter]:
    fence = re.findall(r"```.*?```", text, flags=re.S)
    body = re.sub(r"```.*?```", "", text, flags=re.S)
    code = CODE.findall(body)
    prose = CODE.sub(" ", body)
    rows = [ln.count("|") for ln in body.splitlines() if ln.lstrip().startswith("|")]
    return {"numbers": Counter(n.replace("−", "-") for n in NUM.findall(prose)),
            "code spans": Counter(code), "fenced blocks": Counter(fence),
            "table row shapes": Counter(rows)}


def main() -> int:
    old, new = (open(p).read() for p in sys.argv[1:3])
    a, b = items(old), items(new)
    bad = 0
    for k in a:
        lost, gained = a[k] - b[k], b[k] - a[k]
        n = sum(lost.values()) + sum(gained.values())
        bad += n
        print(f"{k}: {sum(a[k].values())} in the backup, {sum(b[k].values())} in the rewrite, "
              f"{sum(lost.values())} lost, {sum(gained.values())} gained")
        for tag, c in (("LOST", lost), ("GAINED", gained)):
            for item, cnt in c.most_common(25):
                print(f"   {tag} x{cnt}: {item[:100]!r}")
    print("OK" if bad == 0 else f"{bad} differences")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
