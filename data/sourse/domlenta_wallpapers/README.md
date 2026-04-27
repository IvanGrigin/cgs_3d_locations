# Domlenta Wallpapers Catalog

This directory is the expected local location for the separately transferred Domlenta wallpapers catalog.

The directory is intentionally ignored by git except for this placeholder. Put the received archive contents here:

```text
data/sourse/domlenta_wallpapers/
  products.csv
  product_properties.csv
  product_images.csv
  products_characteristics.json
  normalized_wall_materials.jsonl
  images/
```

Refresh normalized wall materials after copying images:

```bash
python3 src/tools/run_wall_material_selector.py normalize \
  --products-csv data/sourse/domlenta_wallpapers/products.csv \
  --out-jsonl data/sourse/domlenta_wallpapers/normalized_wall_materials.jsonl
```
