# ISL Avatar MVP

This is a local, API-backed Indian Sign Language demonstration product. It
plays validated stored pose clips through a browser avatar; it is not a
claim of unrestricted or signer-validated sentence-to-ISL translation.

## Run the product

```powershell
.\start_product.ps1
```

Open `http://127.0.0.1:8080/isl-avatar-prototype.html` in the same computer.
The HTML page requires this API server. Opening the page as a file, or from a
static host without the API, intentionally shows an unavailable state rather
than playing an unrelated fallback gesture.

## Current product surface

- 11,033 locally indexed, browser-ready sign or exact-sentence entries.
- 100 recorded multiword sentence clips selected only on exact matching input.
- Validated lookup and one chosen 30-frame pose clip per resolved entry.
- Reviewed phrase support for English, Hindi, Malayalam, and Tamil.
- Explicit rejection for unreviewed regional-language input.

## Before a demo

```powershell
.\run_product_preflight.ps1
```

These checks establish lookup, API, and pose delivery behavior. They do not
measure linguistic ISL correctness; that requires evaluation by ISL signers.

## Deployment Handoff

The frontend can be hosted as static files, but the API needs a separate
persistent backend because the validated pose runtime is about 2.1 GB. Follow
[deployment/RELEASE_CHECKLIST.md](deployment/RELEASE_CHECKLIST.md) for the
non-secret backend origin, CORS, API-base, and release-gate steps. No public
deployment is created by the repository scripts.

## Release Evidence

Run `./run_release_metrics.ps1` when preparing a release review to refresh the
local lookup-coverage and release-candidate reports. It is intentionally
separate from the faster preflight because it scans the full held corpus.
