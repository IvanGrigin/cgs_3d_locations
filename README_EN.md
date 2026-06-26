# Room Design Generation from Text and Floor Plan

[Русская версия](README.md)

Bachelor thesis project by Ivan Grigin, group M3432, ITMO University. Scientific advisor: Valeria Efimova, PhD, Associate Professor at the Faculty of Information Technologies and Programming.

The project implements automatic generation of interior 3D scenes from a room plan and a natural-language user prompt. The system takes room geometry, interprets design requirements, places furniture, validates constraints, replaces abstract objects with real supplier products, and exports the result as a structured scene, Blender file, renders, and cost estimate.

## Thesis Materials

- [Final thesis text](thesis.pdf)
- [Defense presentation](defense_presentation.pdf)
- [Pipeline diagram](diagram.drawio.png)

## Motivation

Preparing interior 3D scenes requires manual object placement, model selection, material selection, and constraint checking. Image generators produce photorealistic pictures, but they do not create editable 3D scenes with dimensions and coordinates. Online planners support real products and room geometry, but layout creation remains mostly manual. Research 3D generators automate layout synthesis, but often fail to handle the room plan, user requirements, and real supplier products at the same time.

The goal of this work is to reduce interior 3D scene preparation time from hours to minutes while using real manufacturer assets and preserving a target quality level of at least 6/10 on a normalized visual-language model score.

## Implemented Features

- user prompt interpretation into a structured room specification;
- comparison of DiffuScene, M3DLayout, and Infinigen as primary layout generators;
- main layout generation with Infinigen Indoors;
- fast procedural generators for bedrooms, kitchens, bathrooms, toilets, and constrained cases;
- geometric validation: room boundaries, collisions, doors, windows, and walkways;
- scene postprocessing: collision repair, missing object insertion, curtains, and surface materials;
- replacement of generated objects with real supplier products and materials;
- missing GLB asset generation from product photos with TRELLIS.2;
- export to JSON, Blender, renders, report, and cost estimate;
- experimental evaluation of generation time, placement correctness, and visual quality.

## Generation Pipeline

```text
Room geometry + text prompt
    -> requirement interpretation
    -> initial layout with Infinigen or a procedural generator
    -> geometric validation and scene repair
    -> replacement with real products and materials
    -> structured 3D scene, Blender file, renders, and cost estimate
```

Inputs include room geometry with wall height, doors and windows, plus a text prompt describing the room type, style, palette, required objects, and constraints.

Outputs include:

- structured scene description;
- Blender file;
- rendered views;
- selected product list;
- cost estimate with prices and supplier links;
- diagnostic quality and constraint reports.

## Technical Architecture

The system is implemented as an artifact-based pipeline. Each stage not only passes data forward in memory, but also writes intermediate JSON files and reports into the run directory. This makes every run reproducible and debuggable: the input room, prompt interpretation, placement, structured scene, supplier replacements, build report, and VLM scores can be inspected independently.

The central entry point is [src/run_pipeline.py](src/run_pipeline.py). It parses CLI arguments, applies configuration from [src/pipeline_config.py](src/pipeline_config.py), creates run directories, invokes generators through [src/pipeline_runners.py](src/pipeline_runners.py), normalizes artifacts through [src/pipeline_artifacts.py](src/pipeline_artifacts.py), and then runs optional stages: procedural generation, materials, supplier replacement, Blender assembly, rendering, and reporting.

Simplified data flow:

```text
room.json + prompt
  -> room intent / design spec
  -> placement.v1.json
  -> scene.v1.json
  -> repaired scene.v1.json
  -> supplier bindings
  -> supplier_scene.v1.json
  -> scene.blend + renders + reports
```

The implementation keeps different scene representation levels separate. `placement.v1` stores the layout result: object categories, coordinates, dimensions, and rotations. `scene.v1` combines the layout with room geometry, materials, and metadata. `supplier_scene.v1` is produced after abstract objects are replaced with real products and 3D assets.

## Main Modules

| Area | Modules | Purpose |
| --- | --- | --- |
| Orchestration | `src/run_pipeline.py`, `src/pipeline_config.py`, `src/pipeline_runners.py`, `src/pipeline_artifacts.py` | CLI parsing, mode selection, stage execution, artifact normalization and tracking. |
| Prompt analysis | `src/style_prompt_analyzer.py`, `src/pipeline/semantic_room_planner_stage.py`, `src/pipeline/semantic_room_planner/*` | Extract room type, style, palette, zones, objects, relations, and constraints. |
| Initial placement | `src/Plasement/run_infinigen_clean.py`, `src/pipeline_runners.py`, `src/pipeline/procedural_room_stage.py` | Run Infinigen or procedural placement and produce the initial `placement.v1`. |
| Procedural rooms | `src/pipeline/procedural_rooms/*` | Bedroom, living room, corridor, bathroom, and toilet generators with boundary, walkway, and collision checks. |
| Postprocessing | `src/pipeline/infinigen_scene_improvers.py`, `src/pipeline_scene_repair.py`, `src/topview_vlm_orientation_repair.py` | Repair intersections, heights, orientations, and other scene issues. |
| Materials | `src/pipeline/flooring_stage.py`, `src/pipeline/wall_stage.py`, `src/pipeline/curtain_stage.py`, `src/pipeline/kitchen_stage.py` | Select and apply floor, wall, curtain, and kitchen-specific assets. |
| Supplier matching | `src/supplier_layout_matcher.py`, `src/suppliers/*`, `src/apply_supplier_bindings.py` | Filter products, rank candidates, choose replacements, and apply real assets to the scene. |
| Assets | `src/acquire_supplier_bindings_assets.py`, `src/trellis_supplier_asset_orchestrator.py`, `src/tools/blender_supplier_asset.py` | Download, cache, prepare, and generate missing 3D models. |
| Reports and export | `src/supplier_replacement_report.py`, `src/Plasement/blender_scene_builder.py`, `src/tools/blend_to_orbit_gif.py` | Build Blender scenes, renders, GIFs, replacement reports, and pricing reports. |

## Data Formats

Main intermediate formats:

- `room.json` - input geometry: `room.id`, `room.type`, `floor_polygon`, `walls`, `doors`, `windows`, `ceiling_height_m`, coordinate system.
- `room_intent` / `design_spec` - normalized requirements: room type, style, palette, density, desired objects, and constraints.
- `placement.v1.json` - initial placement without the full scene: objects, categories, roles, centers, dimensions, rotations, and placement rules.
- `scene.v1.json` - central pipeline format: room + objects + poses + dimensions + materials + supplier matching targets.
- `supplier_replacements.summary.json` - selected products, prices, links, selection reasons, and alternative candidates.
- `supplier_scene.v1.json` - scene after supplier replacements, ready for Blender assembly.
- `run_manifest.json` - run parameters and produced artifact list.
- `pipeline_stage_timings.json` - per-stage pipeline timings.
- `build_report.json` - diagnostics for model import, scale, materials, and Blender scene assembly.

This separation is important for debugging. If the final scene is wrong, the failing stage can be isolated: prompt interpretation, initial layout, postprocessing, supplier matching, or Blender assembly.

## Prompt Interpretation

The free-form prompt is converted into a unified specification. For example:

```text
A bedroom in dark green tones with wooden furniture and a workplace
```

is converted into JSON with room type, style, density, decor richness, and furniture semantics:

```json
{
  "room_type": "Bedroom",
  "style_label": "contemporary",
  "furniture": [
    {"semantic": "Bed", "count": 1, "priority": "required"},
    {"semantic": "Chair", "count": 1, "priority": "desired"},
    {"semantic": "Table", "count": 1, "priority": "desired"},
    {"semantic": "Storage", "count": 2, "priority": "desired"},
    {"semantic": "Rug", "count": 1, "priority": "desired"},
    {"semantic": "Lighting", "count": 2, "priority": "desired"}
  ],
  "style_raw": "dark green tones with wooden furniture",
  "density": "medium",
  "decor_richness": "low"
}
```

This step uses LLMs through Ollama, including `gpt-oss:20b` and `Mistral 7B`, together with heuristic normalization.

## Layout Generation

DiffuScene, M3DLayout, and Infinigen Indoors were evaluated as primary layout generators. DiffuScene and M3DLayout automate 3D scene synthesis, but they poorly handle exact room boundaries and geometric correctness in this task. Infinigen Indoors works better with constraints and exports editable Blender scenes, so it is used as the main high-quality generator.

Generators were compared against 3D-FRONT with object placement heatmaps and spatial similarity metrics, including Total Variation distance and Jensen-Shannon distance.

## Procedural Generators

Infinigen provides high-quality initial scenes, but remote execution and file transfer can increase full-cycle runtime to 15-20 minutes for heavy scenes. Therefore, fast procedural generators were implemented.

For bedrooms, the system places the main object first, usually the bed, and then adds nightstands, wardrobe, lights, textiles, and decor around it. Each object is checked against room boundaries, collisions, and walkways.

For kitchens, a separate one-wall procedural generator selects a suitable wall, aligns kitchen modules, and checks joints, walkways, appliance opening space, and collisions. Procedural initial placement takes up to 15 seconds.

## Scene Validation And Repair

After initial generation, the system performs postprocessing:

- collision repair;
- insertion of missing objects;
- curtain insertion;
- floor and wall material replacement;
- door, window, and walkway checks;
- procedural replacement for difficult small rooms such as bathrooms and toilets.

This stage improves physical correctness and visual quality and prepares the scene for replacement with real supplier products.

## Real Products And Cost Estimate

The replacement catalog contains:

- 38,899 product cards from 14 sources;
- more than 2,000 manufacturer-provided models;
- furniture, materials, prices, links, and 3D assets;
- VLM-enriched product descriptions from product images.

Product matching uses category, dimensions, style, material, color, price, and 3D model availability. After filtering and ranking, an LLM selects the best product from top-N candidates.

The final report includes furniture dimensions, product photos, item prices, finishing material cost estimates, and links to original supplier products.

## Missing Asset Generation

Many supplier products do not include a reusable 3D model in a suitable format. For such cases, TRELLIS.2 is used to generate a GLB model from the product image and product card. Generated assets are additionally evaluated because a good single view does not guarantee correct geometry, materials, or completeness.

## Quality Evaluation

The minimum target quality level is 6/10 on a normalized metric. For normalization, a VLM evaluates reference designer rooms and real interior photos; generated scenes are then scored against this scale.

Evaluation criteria:

- prompt consistency;
- absence of intersections;
- quality of product and material replacements;
- room coverage by objects;
- placement quality;
- style consistency;
- model confidence;
- overall impression.

The experiment used 65 rooms. Each room was rendered from multiple viewpoints before and after postprocessing.

| Room type | Rooms | VLM before repair | VLM after repair | Average time |
| --- | ---: | ---: | ---: | ---: |
| Bedroom | 23 | 3.54 ± 0.69 | 5.96 ± 0.85 | 8.03 ± 0.64 min |
| Kitchen | 14 | 2.41 ± 1.13 | 5.97 ± 1.33 | 12.69 ± 3.45 min |
| Living room | 22 | 3.19 ± 0.58 | 6.52 ± 0.76 | 6.79 ± 1.01 min |
| Bathrooms | 6 | 2.78 ± 1.00 | 5.81 ± 1.40 | 4.13 ± 0.41 min |
| Average | 65 | 3.14 ± 0.42 | 6.19 ± 0.46 | 7.88 ± 1.13 min |

After collision repair, object insertion, material replacement, and procedural generation for difficult rooms, the normalized score increased by about three points and exceeded the target threshold.

## Technical Organization

Main tools:

- Python;
- Blender Python API;
- Infinigen Indoors;
- Ollama;
- `gpt-oss:20b`, `Mistral 7B`;
- Llama 3.2 Vision;
- TRELLIS.2;
- Pytest.

Some modules run locally, while GPU-heavy stages such as Infinigen execution, VLM evaluation, and asset generation can run on remote servers.

## Repository Structure

```text
config/                  path, policy, and environment configuration
src/                     main pipeline code
  run_pipeline.py        main CLI orchestrator
  pipeline_config.py     run, mode, and path configuration
  pipeline_runners.py    primary generator execution
  pipeline_artifacts.py  placement/scene/blender artifact normalization
src/pipeline/            generation, postprocessing, and export stages
  semantic_room_planner/ semantic planner: zones, objects, relations
src/pipeline/procedural_rooms/
                         procedural room generators
src/suppliers/           supplier catalog, data models, product matching
src/tools/               batch utilities, rendering, reports, converters
tests/                   automated tests
diploma/                 thesis sources and supporting materials
```

## Example Run

```bash
python3 src/run_pipeline.py \
  --room data/input/room_rec_small.json \
  --prompt "Light modern bedroom with a workplace" \
  --placer infinigen_clean \
  --modes infinigen_clean \
  --run-dir out/example_bedroom \
  --procedural-rooms auto \
  --procedural-density very_high \
  --supplier-catalog-json supplier_catalog_canonical.json \
  --supplier-selection-modes cheapest,optimal,best_match \
  --build-supplier-blend \
  --validate-supplier-variants \
  --render \
  --keep-blend
```

## Main Artifacts

- `placement.v1.json` - initial object placement;
- `scene.v1.json` - structured scene merged with room geometry;
- `supplier_scene.v1.json` - scene after replacement with real products;
- `scene.blend` - Blender scene;
- `supplier_replacements.summary.json` - selected products, prices, links, and selection reasons;
- `build_report.json` - diagnostics for import, scale, materials, and assets.

Each run corresponds to one `run-dir`. It stores inputs, prompt analysis results, intermediate JSON files, validation reports, supplier replacement variants, the Blender file, and renders. The final result can therefore be inspected, reproduced, and debugged stage by stage.

## Outcome

The project implements a complete pipeline for room design generation from text and floor plan using real products. Scene preparation time is reduced to 5-20 minutes instead of hours or a full day of manual work. A supplier catalog from Russian manufacturers was collected and extended with generated 3D models, quality evaluation was implemented, and the final result can be exported for further use. The system is being integrated into ITMO's rTIM project for creating and evaluating urban development concepts.
