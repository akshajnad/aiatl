-- StreetSage Database Schema
-- Events and OCR tables

CREATE TABLE IF NOT EXISTS events (
  event_id STRING,
  ts TIMESTAMP_TZ,
  session_id STRING,
  object_class STRING,       -- 'person','bicycle','car','puddle','uneven','cone','text'
  side STRING,               -- 'left','center','right'
  distance_bucket STRING,    -- 'very_near','near','mid','far','unknown'
  ttc_sec FLOAT,             -- NULL if static
  confidence FLOAT,
  instruction STRING,        -- spoken text (if any)
  source STRING              -- 'cv','ocr','rule'
);

CREATE TABLE IF NOT EXISTS ocr_text (
  event_id STRING,
  ts TIMESTAMP_TZ,
  snippet STRING,
  side STRING,
  distance_bucket STRING
);
