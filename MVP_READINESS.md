# ISL Avatar MVP: Current Launch Candidate

## What the local MVP does

- Resolves supported English input against 10,936 server-indexed sign labels.
- Retrieves only the requested recorded pose clips from the local dictionaries.
- Supports 60 reviewed Hindi, Malayalam, and Tamil meeting phrases through deterministic English glosses.
- Rejects unreviewed regional text and unknown English terms instead of repeating a default sign.

## Verified acceptance evidence

- English test set: 25 of 25 phrases resolved to retrievable stored clips.
- Reviewed regional test set: 60 of 60 phrases resolved to retrievable stored clips.
- The ISL500 supplemental source is excluded from serving because its audit
  found both hands collapsed in 9,436 of 14,220 frames. A further five CISLR
  labels and five iSign labels are excluded by the per-clip hand-quality gate.
- Every accepted clip is checked for the expected landmark-array structure before the report passes.

Run `powershell -ExecutionPolicy Bypass -File .\run_mvp_checks.ps1` to reproduce the checks.
Run `isl_env\Scripts\python.exe production_server.py` to serve the MVP at `http://127.0.0.1:8080`.

## Not yet a product claim

This is not unrestricted English-to-ISL translation, nor is it a claim that avatar motion is linguistically validated for every dictionary label. Broad arbitrary Malayalam, Tamil, and Hindi conversation needs a separately evaluated translation corpus and signer review. Public hosting is blocked until the local acceptance suite and avatar visual checks are expanded.
