Local semantic data assets.

- `seed_records.jsonl`: minimal bootstrap dataset for offline development/tests.
- `records.jsonl`, `vectors.npy`: generated index artifacts from `scripts/build_indexes.py`.

Production flow:
1. Prepare local dumps (GeoNames, Wikidata subsets, optional event-venue mappings).
2. Convert to unified JSONL records with fields used by `IndexRecord`.
3. Run `python scripts/build_indexes.py --seed <converted.jsonl> --out data/semantic`.
