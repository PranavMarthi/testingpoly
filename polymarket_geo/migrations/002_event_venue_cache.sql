-- Migration 002: Event venue cache for event-market inference

CREATE TABLE IF NOT EXISTS event_venue_cache (
    id                  BIGSERIAL PRIMARY KEY,
    event_key           TEXT NOT NULL,
    event_year          INTEGER,
    status              TEXT NOT NULL DEFAULT 'not_available', -- confirmed | uncertain | not_available
    venue_name          TEXT,
    city                TEXT,
    country             TEXT,
    latitude            DOUBLE PRECISION,
    longitude           DOUBLE PRECISION,
    geog                GEOGRAPHY(POINT, 4326),
    source_url          TEXT,
    source_type         TEXT DEFAULT 'web',
    confidence          REAL NOT NULL DEFAULT 0.0,
    reason              TEXT,
    raw_payload         JSONB,
    fetched_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at          TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '7 days'),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (event_key, event_year)
);

CREATE INDEX IF NOT EXISTS idx_event_venue_cache_key_year
    ON event_venue_cache (event_key, event_year);

CREATE INDEX IF NOT EXISTS idx_event_venue_cache_expires
    ON event_venue_cache (expires_at);

CREATE INDEX IF NOT EXISTS idx_event_venue_cache_geog
    ON event_venue_cache USING GIST (geog) WHERE geog IS NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname = 'trg_event_venue_cache_updated_at'
    ) THEN
        CREATE TRIGGER trg_event_venue_cache_updated_at
            BEFORE UPDATE ON event_venue_cache
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;
END $$;
