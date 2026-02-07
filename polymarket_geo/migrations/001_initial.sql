-- Migration 001: Initial schema for Polymarket geo pipeline
-- Requires: PostgreSQL 14+ with PostGIS extension

-- Enable PostGIS
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- for fuzzy text search

-- ============================================================
-- MARKETS TABLE
-- Stores raw market data from Polymarket with dedup on condition_id
-- ============================================================
CREATE TABLE IF NOT EXISTS markets (
    id                  BIGSERIAL PRIMARY KEY,
    condition_id        TEXT NOT NULL UNIQUE,  -- Polymarket's unique market identifier
    question            TEXT NOT NULL,
    description         TEXT,
    market_slug         TEXT,
    category            TEXT,
    end_date_iso        TIMESTAMPTZ,
    active              BOOLEAN NOT NULL DEFAULT TRUE,
    closed              BOOLEAN NOT NULL DEFAULT FALSE,
    volume              NUMERIC(20, 2),
    liquidity           NUMERIC(20, 2),
    outcomes            JSONB,                -- ["Yes", "No"] or custom outcomes
    outcome_prices      JSONB,                -- [0.65, 0.35]
    tags                JSONB,                -- array of tag strings
    raw_payload         JSONB,                -- full API response for this market
    -- Pipeline tracking
    geo_processed       BOOLEAN NOT NULL DEFAULT FALSE,
    geo_processed_at    TIMESTAMPTZ,
    geo_version         INTEGER NOT NULL DEFAULT 0,  -- bump when inference logic changes
    -- Timestamps
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for fast lookup by condition_id (already UNIQUE, but explicit)
CREATE INDEX IF NOT EXISTS idx_markets_condition_id ON markets (condition_id);
-- Index for pipeline: find unprocessed markets
CREATE INDEX IF NOT EXISTS idx_markets_unprocessed ON markets (geo_processed, geo_version)
    WHERE geo_processed = FALSE;
-- Index for active markets
CREATE INDEX IF NOT EXISTS idx_markets_active ON markets (active) WHERE active = TRUE;
-- Trigram index for text search on question
CREATE INDEX IF NOT EXISTS idx_markets_question_trgm ON markets USING gin (question gin_trgm_ops);

-- ============================================================
-- MARKET_LOCATIONS TABLE
-- Stores inferred geographic locations per market (1:N relationship)
-- A market can have multiple candidate locations with different confidences
-- ============================================================
CREATE TABLE IF NOT EXISTS market_locations (
    id                  BIGSERIAL PRIMARY KEY,
    market_id           BIGINT NOT NULL REFERENCES markets(id) ON DELETE CASCADE,
    -- Location metadata
    location_name       TEXT NOT NULL,           -- "Atlanta, GA" or "Washington, DC"
    location_type       TEXT NOT NULL DEFAULT 'city',  -- city, state, country, building, arena, global
    confidence          REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    reason              TEXT,                    -- "NLP entity: Atlanta (GPE)" or "Heuristic: Hawks -> Atlanta"
    inference_method    TEXT NOT NULL DEFAULT 'nlp',  -- nlp, heuristic, llm, manual
    -- Geocoded coordinates (nullable until geocoded)
    latitude            DOUBLE PRECISION,
    longitude           DOUBLE PRECISION,
    geog                GEOGRAPHY(POINT, 4326), -- PostGIS geography for spatial queries
    -- Geocoding metadata
    geocoded            BOOLEAN NOT NULL DEFAULT FALSE,
    geocode_source      TEXT,                    -- "nominatim", "google", "manual", "cache"
    geocode_raw         JSONB,                   -- raw geocoder response
    -- Dedup and versioning
    geo_version         INTEGER NOT NULL DEFAULT 1,  -- matches the inference version that created this
    -- Timestamps
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- Prevent duplicate location rows per market per version
    UNIQUE (market_id, location_name, geo_version)
);

-- GIST index on geography for spatial queries (ST_DWithin, ST_Distance)
CREATE INDEX IF NOT EXISTS idx_market_locations_geog ON market_locations USING GIST (geog);
-- Index for lookups by market
CREATE INDEX IF NOT EXISTS idx_market_locations_market_id ON market_locations (market_id);
-- Index for confidence-based ordering
CREATE INDEX IF NOT EXISTS idx_market_locations_confidence ON market_locations (confidence DESC);
-- Composite index for spatial + confidence queries
CREATE INDEX IF NOT EXISTS idx_market_locations_geog_confidence
    ON market_locations USING GIST (geog) WHERE geocoded = TRUE;

-- ============================================================
-- GEOCODE_CACHE TABLE
-- Persistent cache for geocoding results to avoid repeated API calls
-- ============================================================
CREATE TABLE IF NOT EXISTS geocode_cache (
    id                  BIGSERIAL PRIMARY KEY,
    query_normalized    TEXT NOT NULL UNIQUE,     -- normalized location string (lowercase, trimmed)
    latitude            DOUBLE PRECISION,
    longitude           DOUBLE PRECISION,
    display_name        TEXT,                     -- full display name from geocoder
    raw_response        JSONB,
    source              TEXT NOT NULL DEFAULT 'nominatim',
    -- TTL management
    fetched_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at          TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '30 days'),
    hit_count           INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_geocode_cache_query ON geocode_cache (query_normalized);
CREATE INDEX IF NOT EXISTS idx_geocode_cache_expires ON geocode_cache (expires_at);

-- ============================================================
-- PIPELINE_RUNS TABLE
-- Track each pipeline execution for observability
-- ============================================================
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id                  BIGSERIAL PRIMARY KEY,
    started_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at         TIMESTAMPTZ,
    status              TEXT NOT NULL DEFAULT 'running',  -- running, completed, failed
    markets_fetched     INTEGER DEFAULT 0,
    markets_new         INTEGER DEFAULT 0,
    markets_updated     INTEGER DEFAULT 0,
    locations_inferred  INTEGER DEFAULT 0,
    locations_geocoded  INTEGER DEFAULT 0,
    geocode_cache_hits  INTEGER DEFAULT 0,
    geocode_cache_misses INTEGER DEFAULT 0,
    avg_confidence      REAL,
    error_message       TEXT,
    metadata            JSONB
);

-- ============================================================
-- HELPER FUNCTIONS
-- ============================================================

-- Auto-update updated_at on row modification
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER trg_markets_updated_at
    BEFORE UPDATE ON markets
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE OR REPLACE TRIGGER trg_market_locations_updated_at
    BEFORE UPDATE ON market_locations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- USEFUL VIEWS
-- ============================================================

-- View: markets with their best (highest confidence) location
CREATE OR REPLACE VIEW v_markets_best_location AS
SELECT DISTINCT ON (m.id)
    m.id AS market_id,
    m.condition_id,
    m.question,
    m.category,
    m.active,
    ml.location_name,
    ml.location_type,
    ml.confidence,
    ml.latitude,
    ml.longitude,
    ml.geog,
    ml.reason
FROM markets m
LEFT JOIN market_locations ml ON ml.market_id = m.id AND ml.geocoded = TRUE
ORDER BY m.id, ml.confidence DESC NULLS LAST;

-- View: pipeline health metrics
CREATE OR REPLACE VIEW v_pipeline_metrics AS
SELECT
    COUNT(*) AS total_markets,
    COUNT(*) FILTER (WHERE geo_processed) AS processed_markets,
    ROUND(100.0 * COUNT(*) FILTER (WHERE geo_processed) / NULLIF(COUNT(*), 0), 1) AS pct_processed,
    COUNT(*) FILTER (WHERE id IN (SELECT DISTINCT market_id FROM market_locations)) AS markets_with_locations,
    (SELECT ROUND(AVG(confidence)::numeric, 3) FROM market_locations) AS avg_confidence,
    (SELECT COUNT(*) FROM market_locations WHERE confidence < 0.5) AS low_confidence_count
FROM markets;
