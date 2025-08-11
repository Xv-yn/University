# End-to-End Plan

## Phase 0 — Decide the “demo” mode (quick check-in)

- Option A (default/MVP): Batch images & short videos uploaded in the UI.

- Option B (stretch): Live camera feed (USB/IP cam) with near-real-time overlay.

- Ask the sponsor: Is batch OK for the demo, with live feed as a stretch?

## Phase 1 — Target species & keypoint schema

Goal: Fix “what we’re labeling” before touching models.

- Pick one species first (e.g., cattle); list a second (sheep) as stretch.

- Define keypoints (e.g., nose, ears, withers, tail base, knees, hocks, hooves).
  GPT recommends ≤ 18 for speed.

- Define behaviors: standing, lying, walking, feeding. Add “other/unknown”.

- Write a 1-page data schema:

```
image_id, species, bbox, keypoints: [(x,y,vis), ...], behavior, split.
```

- Deliverable: Schema doc + label guide (with example images).
