# ISL Dataset Audit

## Already imported

- CISLR: 4,765 indexed signs used by the browser.
- `vidit031/isl-isolated-40words`: 642 clips, 40 labels; 641 retained after quality filtering.

## Highest-value next sources

| Source | Best use | Status | Decision |
|---|---|---|---|
| `bridgeconn/sign-dictionary-isl` | Isolated vocabulary and landmark variation | CC BY-SA 4.0; about 3,077 glosses; 7.1 GB WebDataset | Import shard-by-shard after storage/time planning; do not mix blindly with CISLR |
| `Exploration-Lab/iSign` | Sentence-level ISL-English translation and text-to-pose research | CC BY-NC-SA 4.0; access requires accepting Hub conditions; about 228 GB | Use for translation training/evaluation, not direct browser dictionary loading |
| ISL fingerspelling dataset | Names, acronyms, out-of-vocabulary terms | 1,308 continuous fingerspelling segments with aligned text | Add after license/annotation review; valuable for the current fallback |
| ISLVT sentence dataset | Sentence-level video/gloss experiments | CC BY 4.0 listing; access and provenance still need verification | Candidate for sentence translation evaluation |

## Import rules

1. Keep source provenance and license beside every converted clip.
2. Split by signer/video, never random frames, to avoid leakage.
3. Deduplicate by source hash and perceptual hash before merging.
4. Keep isolated-sign recognition data separate from sentence-translation data.
5. Do not promote a new sign into browser playback until its hand orientation and signer label are reviewed.

The bridgeconn archive was intentionally not imported in this pass because the full archive is roughly 7.1 GB and the transfer was incomplete. The partial download is not treated as validated data.
