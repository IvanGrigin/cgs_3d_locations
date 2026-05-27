# Restore archived unused files

Manifest: `src/arxiv/archive_manifest_unused_run_pipeline_infinigen_procedural_20260526_130635.json`

Restore command from repo root:

```bash
bash src/arxiv/restore_unused_run_pipeline_infinigen_procedural_20260526_130635.sh
```

The script refuses to overwrite conflicting existing files. If a file was already restored from `src/arxiv` to its original path, the script counts it as `already_restored` and continues.
