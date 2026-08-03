-- Run this once in your Supabase project's SQL editor (Project -> SQL Editor -> New query).
--
-- ALREADY HAVE THE TABLE? Just add the duplicate-detection column:
--   alter table calls add column if not exists content_hash text;
--   create index if not exists calls_content_hash_idx on calls (content_hash);

-- 1. Table that stores every scored call.
create table if not exists calls (
  id bigint generated always as identity primary key,
  filename text not null,          -- Supabase Storage object path for the audio file
  original_name text not null,     -- original uploaded filename, for display
  content_hash text,               -- SHA-256 of the audio, for duplicate detection
  agent_name text,
  call_date date,
  call_topic text,
  status text not null default 'uploaded',   -- uploaded -> transcribing -> transcribed -> scoring -> done | error
  status_message text,
  transcript text,
  scores_json jsonb,
  total_score numeric,
  max_score numeric,
  grade text,
  scoring_method text,
  created_at timestamptz not null default now()
);

create index if not exists calls_created_at_idx on calls (created_at desc);
create index if not exists calls_status_idx on calls (status);
create index if not exists calls_content_hash_idx on calls (content_hash);

-- Row Level Security stays ON with NO policies. The app only ever talks to this
-- table using the service_role key (server-side, in storage.py), which bypasses
-- RLS entirely - so the anon/public key used in the browser has zero access to it.
alter table calls enable row level security;

-- 2. Storage bucket for audio recordings. Kept private; the app uses signed
-- upload URLs (browser -> Storage direct) and the service role key to download
-- files server-side for transcription.
insert into storage.buckets (id, name, public)
values ('call-recordings', 'call-recordings', false)
on conflict (id) do nothing;

-- Optional: cap individual uploads (bytes) and restrict mime types on this bucket.
-- Adjust file_size_limit as needed (e.g. 104857600 = 100MB) via the Storage UI,
-- or:
-- update storage.buckets set file_size_limit = 104857600 where id = 'call-recordings';
