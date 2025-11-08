-- StreetSage Analytics Views

CREATE OR REPLACE VIEW top_hazards AS
SELECT
  DATE_TRUNC('second', ts) AS bucket_ts,
  object_class,
  side,
  distance_bucket,
  MAX(
    COALESCE(1/NULLIF(ttc_sec,0),0)*0.6 +
    CASE distance_bucket
      WHEN 'very_near' THEN 0.3
      WHEN 'near' THEN 0.2
      WHEN 'mid' THEN 0.1
      ELSE 0
    END +
    COALESCE(confidence,0)*0.2
  ) AS risk_score,
  ANY_VALUE(instruction) AS example_instruction
FROM events
WHERE ts > DATEADD('minute', -2, CURRENT_TIMESTAMP())
GROUP BY 1,2,3,4
ORDER BY risk_score DESC;
