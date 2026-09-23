---
pretty_name: ISL Avatar Validated Runtime
license: other
tags:
  - indian-sign-language
  - pose
  - accessibility
---

# ISL Avatar Validated Runtime

This repository contains the browser runtime for the ISL Avatar MVP: a
quality-filtered lookup index and lazy-loaded 30-frame pose shards. It is not a
general sentence-to-ISL model and does not claim signer-validated linguistic
translation. The associated static application refuses phrases with missing
validated terms instead of substituting an unrelated sign.

The runtime excludes the ISL500 source because local geometry checks found
hand-collapse defects unsuitable for the avatar product.
