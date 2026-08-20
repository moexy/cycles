# Restricted annotation inventories

`vlm-teacher-train-content-inventory-2026-08-20.json` freezes the multiset of 343 training-image
contents without storing filenames, directory names, subject IDs, day ordering, or stage labels.

- Content-inventory SHA-256: `6a7def23bcb8640d7694840541c00ec371e0ce0dc864cd5a485d375b8aa15a4f`
- Images: 343
- Total bytes: 168,804,058
- Duplicate-content groups: 0

The inventory was generated in the later audit context, which is not eligible to perform blinded
annotation. Content hashing itself does not use labels. A fresh restricted annotation context should
recompute the inventory with `scripts/freeze_vlm_inventory.py` to a temporary path and require exact
equality with this artifact before displaying any image.
