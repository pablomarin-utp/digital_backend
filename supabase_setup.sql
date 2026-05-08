-- ── 1. Extensión pgvector ────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS vector;

-- ── 2. Tabla de personas ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS people (
    id         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    name       TEXT        NOT NULL UNIQUE,          -- nombre único por persona
    embedding  vector(512) NOT NULL,                 -- embedding medio FaceNet
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Índice coseno para búsqueda rápida (útil con muchas personas)
CREATE INDEX IF NOT EXISTS people_embedding_idx
    ON people USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 10);

-- ── 3. Tabla de fotos (una o más por persona) ─────────────────────────────────
CREATE TABLE IF NOT EXISTS face_photos (
    id         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    person_id  UUID        NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    storage_path TEXT      NOT NULL,   -- ruta dentro del bucket "faces"
    created_at TIMESTAMPTZ DEFAULT now()
);

-- ── 4. RPC: comparación vectorial ────────────────────────────────────────────
-- Recibe un embedding de consulta y devuelve el mejor match sobre el umbral.
CREATE OR REPLACE FUNCTION match_face(
    query_embedding vector(512),
    similarity_threshold float DEFAULT 0.70,
    match_count int DEFAULT 1
)
RETURNS TABLE (
    person_id  UUID,
    name       TEXT,
    similarity float
)
LANGUAGE sql STABLE
AS $$
    SELECT
        id          AS person_id,
        name,
        1 - (embedding <=> query_embedding) AS similarity
    FROM people
    WHERE 1 - (embedding <=> query_embedding) >= similarity_threshold
    ORDER BY embedding <=> query_embedding   -- orden coseno ascendente = más similar primero
    LIMIT match_count;
$$;

-- ── 5. Trigger: actualiza updated_at automáticamente ─────────────────────────
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN NEW.updated_at = now(); RETURN NEW; END;
$$;

CREATE OR REPLACE TRIGGER people_updated_at
    BEFORE UPDATE ON people
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ── 6. Bucket de Storage para fotos ──────────────────────────────────────────
-- Ejecutar desde el dashboard Storage > New bucket, o via API:
-- INSERT INTO storage.buckets (id, name, public) VALUES ('faces', 'faces', false);
