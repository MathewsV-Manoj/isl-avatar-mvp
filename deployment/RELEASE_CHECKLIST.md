# Deployment Handoff Checklist

This checklist prepares the local MVP for a later split deployment. It does
not publish any service.

## Before Provisioning

1. Run `./run_product_preflight.ps1` from the repository root and retain its
   output with the release notes.
2. Confirm the backend host has at least 3 GB of persistent storage available
   for the current 2.1 GB pose-shard bundle plus logs and headroom.
3. Choose the API hostname before configuring the frontend. The API must serve
   HTTPS in its deployed environment.

## Free Static Deployment

The release also supports a no-server path using a public Hugging Face dataset
runtime plus a Static Space. Run `export_static_runtime.py`, then
`prepare_static_release.py`, and finally `deploy_hf_static.py` while logged in
with a Hugging Face token that has write permission. The script uploads the
2.1 GB validated runtime shards and publishes the static app that lazy-loads
only the selected sign clips.

## Backend Configuration

1. Copy `deployment/api.env.example` into the backend's private environment.
2. Set `ISL_HOST=0.0.0.0` and the provider's `PORT` value.
3. Set `ISL_ALLOWED_ORIGIN` to the exact HTTPS Vercel frontend origin. Do not
   use a wildcard origin for this browser API.
4. Start `production_server.py` with the complete runtime data bundle mounted
   beside it. Check `GET /api/health` from the host's private health checker.

## Frontend Configuration

1. Set the `isl-api-base` meta tag in `isl-avatar-prototype.html` to the API
   HTTPS origin, with no trailing slash.
2. Deploy the static files to Vercel using the included `vercel.json`.
3. Verify a normal lookup, a recorded exact sentence, a reviewed Hindi,
   Malayalam, and Tamil input, and an unresolved word. An unresolved word must
   remain unavailable; it must never play an unrelated sign.

## Release Boundary

The shipped surface is a validated lookup and recorded-pose MVP. It is not a
claim of universal vocabulary or automatic semantic ISL translation. Obtain
independent ISL signer review before describing it as linguistically validated
or using it for accessibility-critical communication.
