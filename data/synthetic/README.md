# Synthetic Data

This folder stores generated synthetic videos and deterministic detection logs for local regression testing.

Generate assets with:

```bash
python src/main.py --generate-synthetic --output outputs
```

Synthetic videos are ignored by git. Detection JSON files are deterministic and can be regenerated.
