-- EngineIQ — anomaly_logs schema migration
-- Run this once in the Supabase SQL editor (Project → SQL Editor → New query)
-- before running inference.py. Without it, log_to_supabase() will fail on
-- every insert (caught internally, so inference.py keeps running — you'll
-- just see "Supabase log failed" warnings and no rows will appear in the
-- table or in index.html's live monitor).
--
-- WHY: inference.py now logs every scored window (not just anomalies), and
-- distinguishes AI4I-dataset replay windows ('simulation', no physical
-- vibration axes) from real ESP32 readings ('live', has accX/accY/accZ/temp).
-- The old schema only had columns for the 'live' shape.

alter table anomaly_logs
  add column if not exists source text,
  add column if not exists raw_window jsonb,
  add column if not exists is_anomaly boolean;

-- acc_x_mean / acc_y_mean / acc_z_mean / temp_mean are only populated for
-- source = 'live' now (they're meaningless for AI4I replay windows), so
-- they can no longer be required on every row.
alter table anomaly_logs
  alter column acc_x_mean drop not null,
  alter column acc_y_mean drop not null,
  alter column acc_z_mean drop not null,
  alter column temp_mean  drop not null;

-- ── ROW LEVEL SECURITY ────────────────────────────────────────────────────
-- inference.py writes with the public anon key. index.html's monitor also
-- reads with the anon key, but only after a user signs in via Supabase Auth
-- (see dashboard.py / index.html's #monitor login gate) — RLS is what
-- actually enforces that, the login form alone does not restrict the API.
-- Adjust to taste; this is the minimum for the app to work as built:
alter table anomaly_logs enable row level security;

drop policy if exists "anon can insert" on anomaly_logs;
create policy "anon can insert" on anomaly_logs
  for insert to anon
  with check (true);

drop policy if exists "authenticated can read" on anomaly_logs;
create policy "authenticated can read" on anomaly_logs
  for select to authenticated
  using (true);
