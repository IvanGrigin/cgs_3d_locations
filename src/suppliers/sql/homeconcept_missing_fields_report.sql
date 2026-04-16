-- This report inspects HomeConcept catalog completeness in the supplier DB.
-- It focuses on missing fields, complete-card counts, and recent daily totals.
-- The queries are meant for manual QA rather than automated monitoring.
-- Output is optimized for sqlite3 terminal usage.
-- Run:
-- sqlite3 -header -column data/sourse/suppliers/suppliers.db < src/suppliers/sql/homeconcept_missing_fields_report.sql

.print ''
.print '1. Daily totals'
SELECT substr(parsed_at, 1, 10) AS parsed_date, COUNT(*) AS records
FROM supplier_product
WHERE source_site = 'homeconcept'
GROUP BY parsed_date
ORDER BY parsed_date DESC;

.print ''
.print '2. Minimum complete card criteria for 2026-04-09'
SELECT
    COUNT(*) AS complete_cards
FROM supplier_product
WHERE source_site = 'homeconcept'
  AND substr(parsed_at, 1, 10) = '2026-04-09'
  AND title IS NOT NULL AND title != ''
  AND product_url IS NOT NULL AND product_url != ''
  AND price_value IS NOT NULL
  AND description IS NOT NULL AND description != ''
  AND width_cm IS NOT NULL
  AND depth_cm IS NOT NULL
  AND height_cm IS NOT NULL
  AND model_download_url IS NOT NULL AND model_download_url != '';

.print ''
.print '3. Missing fields summary for 2026-04-09'
WITH today AS (
    SELECT *
    FROM supplier_product
    WHERE source_site = 'homeconcept'
      AND substr(parsed_at, 1, 10) = '2026-04-09'
)
SELECT
    COUNT(*) AS total,
    SUM(CASE WHEN title IS NULL OR title = '' THEN 1 ELSE 0 END) AS missing_title,
    SUM(CASE WHEN product_url IS NULL OR product_url = '' THEN 1 ELSE 0 END) AS missing_product_url,
    SUM(CASE WHEN price_value IS NULL THEN 1 ELSE 0 END) AS missing_price,
    SUM(CASE WHEN description IS NULL OR description = '' THEN 1 ELSE 0 END) AS missing_description,
    SUM(CASE WHEN width_cm IS NULL THEN 1 ELSE 0 END) AS missing_width,
    SUM(CASE WHEN depth_cm IS NULL THEN 1 ELSE 0 END) AS missing_depth,
    SUM(CASE WHEN height_cm IS NULL THEN 1 ELSE 0 END) AS missing_height,
    SUM(CASE WHEN model_download_url IS NULL OR model_download_url = '' THEN 1 ELSE 0 END) AS missing_model_url
FROM today;

.print ''
.print '4. Structural reasons that can be fixed in parser'
WITH today AS (
    SELECT *
    FROM supplier_product
    WHERE source_site = 'homeconcept'
      AND substr(parsed_at, 1, 10) = '2026-04-09'
)
SELECT
    SUM(CASE WHEN width_cm IS NULL AND raw_html LIKE '%Диаметр%' THEN 1 ELSE 0 END) AS width_missing_but_has_diameter,
    SUM(CASE WHEN depth_cm IS NULL AND raw_html LIKE '%Диаметр%' THEN 1 ELSE 0 END) AS depth_missing_but_has_diameter,
    SUM(CASE WHEN depth_cm IS NULL AND raw_html LIKE '%Длина%' THEN 1 ELSE 0 END) AS depth_missing_but_has_length,
    SUM(CASE WHEN (description IS NULL OR description = '') AND raw_html LIKE '%property="og:description"%' THEN 1 ELSE 0 END) AS missing_description_but_has_og,
    SUM(CASE WHEN price_value IS NULL AND raw_html LIKE '%lowPrice%' THEN 1 ELSE 0 END) AS missing_price_but_has_jsonld,
    SUM(CASE WHEN raw_html LIKE '%<title>404</title>%' THEN 1 ELSE 0 END) AS dead_product_pages
FROM today;

.print ''
.print '5. Incomplete cards and exact missing fields'
WITH today AS (
    SELECT *
    FROM supplier_product
    WHERE source_site = 'homeconcept'
      AND substr(parsed_at, 1, 10) = '2026-04-09'
)
SELECT
    product_url,
    title,
    CASE WHEN title IS NULL OR title = '' THEN 'title ' ELSE '' END ||
    CASE WHEN price_value IS NULL THEN 'price ' ELSE '' END ||
    CASE WHEN description IS NULL OR description = '' THEN 'description ' ELSE '' END ||
    CASE WHEN width_cm IS NULL THEN 'width ' ELSE '' END ||
    CASE WHEN depth_cm IS NULL THEN 'depth ' ELSE '' END ||
    CASE WHEN height_cm IS NULL THEN 'height ' ELSE '' END ||
    CASE WHEN model_download_url IS NULL OR model_download_url = '' THEN 'model_url ' ELSE '' END AS missing_fields
FROM today
WHERE NOT (
    title IS NOT NULL AND title != ''
    AND product_url IS NOT NULL AND product_url != ''
    AND price_value IS NOT NULL
    AND description IS NOT NULL AND description != ''
    AND width_cm IS NOT NULL
    AND depth_cm IS NOT NULL
    AND height_cm IS NOT NULL
    AND model_download_url IS NOT NULL AND model_download_url != ''
)
ORDER BY product_url;
