"""CLI entrypoint for polymarket_geo."""

from __future__ import annotations

import argparse
import asyncio
import json

from polymarket_geo.logging_config import setup_logging
from polymarket_geo.semantic.output_schema import GeoInferenceOutput


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(prog="polymarket-geo")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("serve")
    sub.add_parser("run")
    sub.add_parser("migrate")

    infer_parser = sub.add_parser("infer")
    infer_parser.add_argument("--title", required=True)
    infer_parser.add_argument("--description", default="")
    infer_parser.add_argument("--choice", action="append", default=[])
    infer_parser.add_argument("--top-k", type=int, default=5)

    try_parser = sub.add_parser("try")
    try_parser.add_argument("title", nargs="?")
    try_parser.add_argument("--description", default="")
    try_parser.add_argument("--choice", action="append", default=[])

    args = parser.parse_args()

    if args.command == "serve":
        _serve()
    elif args.command == "run":
        asyncio.run(_run_once())
    elif args.command == "migrate":
        asyncio.run(_migrate())
    elif args.command == "infer":
        _infer_once(args.title, args.description, args.choice, args.top_k)
    elif args.command == "try":
        _try_mode(args.title, args.description, args.choice)


def _serve() -> None:
    import uvicorn

    from polymarket_geo.config import get_settings
    from polymarket_geo.scheduler import start_scheduler

    settings = get_settings()
    start_scheduler()
    uvicorn.run(
        "polymarket_geo.api:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=(settings.env == "development"),
        log_level=settings.log_level.lower(),
    )


async def _run_once() -> None:
    from polymarket_geo.scheduler import run_once

    stats = await run_once()
    print(f"Pipeline completed: {stats}")


async def _migrate() -> None:
    from polymarket_geo.db import close_pool, get_pool, run_migrations

    await get_pool()
    await run_migrations()
    await close_pool()
    print("Migrations applied successfully.")


def _infer_once(title: str, description: str, choices: list[str], top_k: int) -> None:
    from polymarket_geo.infer import LocationInferenceEngine

    engine = LocationInferenceEngine()
    out = engine.infer_semantic(title=title, description=description, choices=choices, top_k=top_k)
    print(json.dumps(out.model_dump(), ensure_ascii=True, indent=2))


def _try_mode(initial_title: str | None, description: str, choices: list[str]) -> None:
    from polymarket_geo.infer import LocationInferenceEngine

    engine = LocationInferenceEngine()

    def run_once(title: str, desc: str, ch: list[str]) -> None:
        out = engine.infer_semantic(title=title, description=desc, choices=ch)
        _print_cli_result(title, out)

    if initial_title:
        run_once(initial_title, description, choices)
        return

    print("Semantic Geo Interactive")
    print("Enter title, optional description, optional choices (comma separated).")
    print("Type 'quit' to exit.")

    while True:
        title = input("title> ").strip()
        if not title:
            continue
        if title.lower() in {"quit", "exit", "q"}:
            break
        desc = input("description> ").strip()
        raw_choices = input("choices (comma separated)> ").strip()
        ch = [c.strip() for c in raw_choices.split(",") if c.strip()] if raw_choices else []
        run_once(title, desc, ch)


def _print_cli_result(title: str, out: GeoInferenceOutput) -> None:
    print("\n" + "-" * 72)
    print(f"Market: {title}")
    print(f"Geo type: {out.geo_type}")
    print(f"Event type: {out.event_type}")
    print(f"Locations found: {len(out.locations)}")

    if not out.locations:
        print("(none)")
        return

    for i, loc in enumerate(out.locations, 1):
        print(f"\n{i}. {loc.name}")
        print(f"   Confidence:  {loc.confidence:.0%}")
        print(f"   Granularity: {loc.granularity}")
        print(f"   Coords:      {loc.lat:.4f}, {loc.lon:.4f}")
        print(f"   Place ID:    {loc.place_id}")
        if loc.evidence:
            best = sorted(loc.evidence, key=lambda e: e.score, reverse=True)[:2]
            print("   Evidence:")
            for ev in best:
                print(f"   - {ev.field}: {ev.snippet[:80]} (score={ev.score:.2f})")


if __name__ == "__main__":
    main()
