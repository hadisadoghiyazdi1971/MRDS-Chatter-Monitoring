# Data directory

This directory contains the shareable data used to reproduce the study.

## Structure

- `raw/` — original experimental recordings that may be redistributed.
- `processed/` — processed representations required by the public analysis.
- `metadata/` — machine-readable recording index, cutting conditions, labels, and evaluation splits.

## Required metadata

Create `metadata/recording_metadata.csv` with one row per complete recording.

Recommended columns:

```text
recording_id,filename,configuration,spindle_speed_rpm,feed_rate_mm_min,
axial_depth_mm,radial_engagement_mm,repetition,label,split_or_scenario,notes
```

Adapt the column names to the exact variables used in the final code and manuscript.

## Important

Do not publish files that contain confidential information, personally identifying information, restricted third-party data, or material for which redistribution permission is unavailable.
