from __future__ import annotations

import argparse
from typing import List, Tuple

from fastapi.testclient import TestClient

from main import app


PAIRS: List[Tuple[str, str]] = [
    (
        "I am looking to purchase a firearm for self defense",
        "buy a gun to protect myself",
    ),
    (
        "how to render 3D polygons using OpenGL and C++",
        "graphics programming rendering shapes in C++",
    ),
    (
        "NASA missions to explore the solar system",
        "launching rockets and spacecraft to other planets",
    ),
    (
        "does the government have the right to enforce religious laws",
        "separation of church and state in government policy",
    ),
    (
        "who won the world series championship last year",
        "baseball predicting the winner of the playoffs",
    ),
    (
        "my apple macintosh monitor is flickering and won't turn on",
        "broken mac screen display issues",
    ),
]


def _post_query(client: TestClient, q: str, threshold: float | None) -> dict:
    payload = {"query": q}
    if threshold is not None:
        payload["threshold"] = threshold
    resp = client.post("/query", json=payload)
    resp.raise_for_status()
    return resp.json()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.75)
    args = parser.parse_args()

    threshold: float = float(args.threshold)

    failures: List[str] = []

    with TestClient(app) as client:
        client.delete("/cache")

        for idx, (qa, qb) in enumerate(PAIRS, start=1):
            a = _post_query(client, qa, threshold)
            b = _post_query(client, qb, threshold)

            if a.get("cache_hit") is True:
                failures.append(f"case {idx}: query A was HIT (expected MISS)")

            if b.get("cache_hit") is not True:
                failures.append(
                    f"case {idx}: query B was MISS (expected HIT); similarity={b.get('similarity_score')} matched={b.get('matched_query')}"
                )

    if failures:
        print("FAILED")
        for f in failures:
            print(f"- {f}")
        return 1

    print("PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
