"""
CLI entrypoint for the Polymarket Geo pipeline.

Usage:
    python -m polymarket_geo serve       # Start API server + scheduler
    python -m polymarket_geo run         # Run pipeline once and exit
    python -m polymarket_geo migrate     # Run DB migrations only
    python -m polymarket_geo infer-test  # Test inference on sample texts
    python -m polymarket_geo try         # Interactive: type a title, get geo data
    python -m polymarket_geo try "title" # One-shot: infer geo for a single title
"""

from __future__ import annotations

import asyncio
import sys

from polymarket_geo.logging_config import setup_logging


def main():
    setup_logging()

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "serve":
        _serve()
    elif command == "run":
        asyncio.run(_run_once())
    elif command == "migrate":
        asyncio.run(_migrate())
    elif command == "infer-test":
        _infer_test()
    elif command == "try":
        _try_interactive()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


def _serve():
    """Start FastAPI server with embedded scheduler."""
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


async def _run_once():
    """Run the full pipeline once."""
    from polymarket_geo.scheduler import run_once
    stats = await run_once()
    print(f"Pipeline completed: {stats}")


async def _migrate():
    """Run database migrations."""
    from polymarket_geo.db import close_pool, get_pool, run_migrations
    await get_pool()
    await run_migrations()
    await close_pool()
    print("Migrations applied successfully.")


def _infer_test():
    """Test the inference engine on sample market texts."""
    from polymarket_geo.infer import LocationInferenceEngine

    engine = LocationInferenceEngine()

    samples = [
        ("Highest temperature in Atlanta on February 7?", None),
        ("Atlanta Hawks vs Lakers", None),
        ("What will Trump say this week?", None),
        ("Will the Fed cut rates?", "The Federal Reserve is expected to announce its decision on interest rates."),
        ("BTC price above 100k by Friday?", None),
        ("Will Russia invade another country?", None),
        ("Super Bowl LVIII winner?", "The game will be played at Allegiant Stadium in Las Vegas."),
        ("Manchester United vs Arsenal Premier League match", None),
        ("Will Congress pass the spending bill?", None),
        ("Earthquake in California this month?", None),
    ]

    print("\n" + "=" * 80)
    print("INFERENCE ENGINE TEST")
    print("=" * 80)

    for question, description in samples:
        result = engine.infer("test-id", question, description)
        print(f"\n{'─' * 60}")
        print(f"Q: {question}")
        if description:
            print(f"D: {description[:80]}...")
        print(f"Global: {result.is_global} | Has Location: {result.has_location}")
        for loc in result.locations:
            print(f"  → {loc.location_name:30s} "
                  f"conf={loc.confidence:.2f}  "
                  f"type={loc.location_type.value:10s}  "
                  f"method={loc.inference_method.value}")
            print(f"    reason: {loc.reason[:100]}")

    print(f"\n{'=' * 80}")


def _try_interactive():
    """
    Interactive mode: type a market title and instantly see inferred geo data.
    Also supports one-shot: python -m polymarket_geo try "Will the Fed cut rates?"
    """
    import logging
    logging.disable(logging.WARNING)  # suppress spacy/config warnings for clean output

    from polymarket_geo.infer import LocationInferenceEngine

    engine = LocationInferenceEngine()

    def _print_result(title: str):
        result = engine.infer("interactive", title)
        has_not_available = any(loc.location_name == "not_available" for loc in result.locations)

        print()
        if has_not_available:
            print("  Type:   Event venue not available yet")
        elif result.is_global and not result.has_location:
            print(f"  Type:   Global (no specific geography)")
        elif result.has_location:
            print(f"  Type:   Location-specific")
        else:
            print(f"  Type:   Unknown / no locations inferred")

        print(f"  Locations found: {len(result.locations)}")

        if not result.locations:
            print("  (none)")
            return

        print()
        for i, loc in enumerate(result.locations, 1):
            print(f"  {i}. {loc.location_name}")
            print(f"     Confidence:  {loc.confidence:.0%}")
            print(f"     Type:        {loc.location_type.value}")
            print(f"     Method:      {loc.inference_method.value}")
            if loc.latitude is not None and loc.longitude is not None:
                print(f"     Coords:      {loc.latitude:.4f}, {loc.longitude:.4f}")
            print(f"     Reason:      {loc.reason}")
            print()

    # One-shot mode: python -m polymarket_geo try "some title here"
    if len(sys.argv) > 2:
        title = " ".join(sys.argv[2:])
        print(f"\n  Market: {title}")
        _print_result(title)
        return

    # Interactive REPL
    print()
    print("  Polymarket Geo - Interactive Tester")
    print("  ====================================")
    print("  Type a prediction market title and press Enter.")
    print("  Type 'quit' or Ctrl+C to exit.")
    print()

    while True:
        try:
            title = input("  > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n")
            break

        if not title:
            continue
        if title.lower() in ("quit", "exit", "q"):
            break

        print(f"\n  Market: {title}")
        _print_result(title)
        print(f"  {'─' * 50}")


if __name__ == "__main__":
    main()
