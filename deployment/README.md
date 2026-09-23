# Deployment Preparation

This repository is prepared for a split deployment, but nothing in this
folder publishes the application.

## Frontend

Host `isl-avatar-prototype.html` as a static site. Before deployment, set the
`isl-api-base` meta tag to the HTTPS origin of the deployed API, for example:

```html
<meta name="isl-api-base" content="https://api.example.com">
```

Vercel is suitable for this static frontend.
The included `vercel.json` rewrites the root URL to the avatar page; it does
not deploy the Python API.

## API and media backend

Run `production_server.py` on a persistent backend with the full project
runtime data available: `signal_dictionary_*`, `reports/clip_eligibility.json`,
`regional_translation.py`, and `production_server.py`. Use the non-secret
values in `api.env.example`, replacing the frontend origin with the real Vercel
origin. The backend must support persistent disk or mounted object storage for
the pose shards; the current runtime pose-shard bundle is about 2.1 GB. Vercel
Functions are not the correct host for this dataset.

## Release gate

Run `run_product_preflight.ps1` locally before any deployment. The release
remains a validated-dictionary MVP until independent ISL signer evaluation is
completed.

`release_readiness_check.py` is included in that preflight and verifies that
every shard referenced by the product manifest is present in the backend
runtime bundle.
