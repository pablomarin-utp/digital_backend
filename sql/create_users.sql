-- SQL para crear la tabla users en Supabase (Postgres)
-- Ejecutar en SQL Editor de Supabase

-- Si quieres usar pgvector, crea la extensión primero (opcional):
-- CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS public.users (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  username text UNIQUE NOT NULL,
  password_hash text NOT NULL,
  full_name text,
  created_at timestamptz DEFAULT now(),
  embedding double precision[]
);
