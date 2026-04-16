-- This report inspects 3ddd catalog completeness in the supplier DB.
-- It summarizes missing critical fields on the latest deduplicated cards.
-- The queries help validate parser quality without downloading models.
-- Output is optimized for sqlite3 terminal usage.
-- Run:
-- sqlite3 -header -column data/sourse/suppliers/suppliers.db < src/suppliers/sql/three_ddd_missing_fields_report.sql

DROP VIEW IF EXISTS latest_three_ddd;
CREATE TEMP VIEW latest_three_ddd AS
SELECT sp.*
FROM supplier_product sp
JOIN (
    SELECT
        COALESCE(NULLIF(product_url, ''), unique_key) AS identity_key,
        MAX(id) AS max_id
    FROM supplier_product
    WHERE source_site = '3ddd'
    GROUP BY COALESCE(NULLIF(product_url, ''), unique_key)
) dedup
  ON sp.id = dedup.max_id;

.print ''
.print '1. Totals'
SELECT
    COUNT(*) AS total,
    SUM(CASE WHEN title IS NULL OR title = '' THEN 1 ELSE 0 END) AS missing_title,
    SUM(CASE WHEN product_url IS NULL OR product_url = '' THEN 1 ELSE 0 END) AS missing_product_url,
    SUM(CASE WHEN price_value IS NULL THEN 1 ELSE 0 END) AS missing_price,
    SUM(CASE WHEN description IS NULL OR description = '' THEN 1 ELSE 0 END) AS missing_description,
    SUM(CASE WHEN width_cm IS NULL THEN 1 ELSE 0 END) AS missing_width,
    SUM(CASE WHEN depth_cm IS NULL THEN 1 ELSE 0 END) AS missing_depth,
    SUM(CASE WHEN height_cm IS NULL THEN 1 ELSE 0 END) AS missing_height,
    SUM(CASE WHEN model_page_url IS NULL OR model_page_url = '' THEN 1 ELSE 0 END) AS missing_model_page,
    SUM(CASE WHEN images_json IS NULL OR images_json = '[]' THEN 1 ELSE 0 END) AS missing_images
FROM latest_three_ddd;

.print ''
.print '2. Minimum complete card criteria'
SELECT
    COUNT(*) AS complete_cards
FROM latest_three_ddd
WHERE title IS NOT NULL AND title != ''
  AND product_url IS NOT NULL AND product_url != ''
  AND price_value IS NOT NULL
  AND description IS NOT NULL AND description != ''
  AND width_cm IS NOT NULL
  AND depth_cm IS NOT NULL
  AND height_cm IS NOT NULL
  AND model_page_url IS NOT NULL AND model_page_url != '';

.print ''
.print '3. Incomplete cards and exact missing fields'
SELECT
    product_url,
    title,
    CASE WHEN title IS NULL OR title = '' THEN 'title ' ELSE '' END ||
    CASE WHEN price_value IS NULL THEN 'price ' ELSE '' END ||
    CASE WHEN description IS NULL OR description = '' THEN 'description ' ELSE '' END ||
    CASE WHEN width_cm IS NULL THEN 'width ' ELSE '' END ||
    CASE WHEN depth_cm IS NULL THEN 'depth ' ELSE '' END ||
    CASE WHEN height_cm IS NULL THEN 'height ' ELSE '' END ||
    CASE WHEN model_page_url IS NULL OR model_page_url = '' THEN 'model_page ' ELSE '' END ||
    CASE WHEN images_json IS NULL OR images_json = '[]' THEN 'images ' ELSE '' END AS missing_fields
FROM latest_three_ddd
WHERE NOT (
    title IS NOT NULL AND title != ''
    AND product_url IS NOT NULL AND product_url != ''
    AND price_value IS NOT NULL
    AND description IS NOT NULL AND description != ''
    AND width_cm IS NOT NULL
    AND depth_cm IS NOT NULL
    AND height_cm IS NOT NULL
    AND model_page_url IS NOT NULL AND model_page_url != ''
)
ORDER BY product_url;
