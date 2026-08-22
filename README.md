# MRDS: Meta-Rényi Distributional Sampling for Recording-Level Chatter Monitoring

[![Status](https://img.shields.io/badge/status-research%20code-blue)](#)
[![Data](https://img.shields.io/badge/data-available%20in%20repository-green)](#data)
[![Code](https://img.shields.io/badge/code-reproducibility%20materials-green)](#code)

## Overview

This repository provides the data, source code, metadata, figures, and reproducibility materials associated with **Meta-Rényi Distributional Sampling (MRDS)**, a recording-level data-reduction framework developed for machine-tool chatter monitoring.

MRDS represents each complete experimental recording through the empirical distribution of descriptors extracted from its local windows. The collection of recording distributions is then treated at a second, meta-distribution level. A finite Rényi-based objective, together with Wasserstein geometry, is used to select a compact, duplicate-free subset of **complete observed recordings** intended to preserve the diversity of the original training data.

The repository is maintained independently of any specific journal. It is intended to support transparent review, reproducibility, reuse, and citation of the research materials.

## Repository contents

```text
.
├── README.md
├── CITATION.cff
├── DATA_CODE_AVAILABILITY.md
├── RELEASE_CHECKLIST.md
├── LICENSES.md
│
├── code/
│   ├── preprocessing/        # Signal preparation and windowing
│   ├── feature_extraction/   # Extraction of recording/window descriptors
│   ├── mrds/                 # MRDS and refined MRDS implementation
│   ├── baselines/            # Comparison/reduction methods
│   ├── evaluation/           # Classification and evaluation scenarios
│   └── utils/                # Shared helper functions
│
├── data/
│   ├── raw/                  # Original shareable recordings
│   ├── processed/            # Processed data used by the experiments
│   ├── metadata/             # Labels, cutting conditions, splits, etc.
│   └── README.md
│
├── results/
│   ├── tables/               # Machine-readable result tables
│   └── figures/              # Reproduced result figures
│
├── assets/
│   ├── experimental_setup.jpg
│   ├── machining_setup.jpg
│   ├── sensor_configuration.jpg
│   └── mrds_workflow.png
│
└── docs/
    └── REPRODUCIBILITY.md
```

> **Note:** The image names shown above are recommended names. Replace the placeholders in `assets/` with the final, publication-safe images before release.

## Experimental data

The study uses experimental machining recordings collected under multiple cutting conditions and repeated trials. The public data package should contain only material that can legally and ethically be redistributed.

For each recording, the accompanying metadata should identify, where applicable:

- recording identifier;
- workpiece/tool configuration;
- spindle speed;
- feed rate;
- axial and radial depth/engagement parameters;
- repetition number;
- ground-truth chatter state or class label;
- sampling information;
- train/test scenario membership.

The machine-readable metadata should be placed in `data/metadata/`.

### Recommended metadata file

A single file such as:

```text
data/metadata/recording_metadata.csv
```

is recommended as the authoritative index linking recording filenames to experimental conditions and labels.

## Data organization

Place the original shareable recordings in:

```text
data/raw/
```

Place any processed representations required to reproduce the reported experiments in:

```text
data/processed/
```

Do not include temporary files, local caches, absolute-path configuration files, private notes, reviewer correspondence, manuscript source files intended only for submission, or copyrighted third-party material.

## Code

The public code is organized by function:

- `code/preprocessing/`: loading, cleaning, segmentation, and sliding-window preparation;
- `code/feature_extraction/`: computation of the descriptors used to represent local windows and recordings;
- `code/mrds/`: finite MRDS subset selection and refinement;
- `code/baselines/`: competing data-reduction or selection approaches;
- `code/evaluation/`: downstream classification and condition-disjoint evaluation;
- `code/utils/`: common utility functions.

The downstream evaluation should use the same settings reported in the manuscript. Any random process should use explicitly documented seeds.

## Reproducing the experiments

A clean public release should allow a reader to reproduce the main results using relative paths only.

Recommended execution order:

```text
1. Prepare / verify metadata
2. Preprocess the recordings
3. Extract window-level descriptors
4. Construct recording-level empirical distributions
5. Run MRDS at the required retention levels
6. Run baseline selection methods
7. Train and evaluate the downstream classifier
8. Export tables and figures
```

Exact commands and software requirements should be documented in `docs/REPRODUCIBILITY.md` after the final public scripts have been placed in this repository.

## Evaluation

The repository should reproduce the condition-disjoint evaluation scenarios described in the manuscript, including:

- critical stability-boundary generalization;
- unseen spindle-speed generalization;
- workpiece/overhang generalization.

Report the same performance metrics and retention levels used in the manuscript, and save machine-readable outputs under `results/tables/`.

## Figures and experimental setup

Only the **experimental setup photographs** from the article should be included in this public repository. Do not include manuscript figures unrelated to the equipment/setup.

Place the selected publication-safe setup image(s) in:

```text
assets/
```

Recommended file name:

```text
assets/experimental_setup.jpg
```

Use the following caption in the README:

```markdown
![Experimental setup for end milling and acoustic signal acquisition](assets/experimental_setup.jpg)

*Experimental setup for end milling and acoustic signal acquisition.*
```

If more than one setup photograph is needed, keep them limited to device/setup views only and use consistent naming such as:

```text
assets/experimental_setup_1.jpg
assets/experimental_setup_2.jpg
```

Before publishing the images, remove or crop any personally identifying information, laboratory access information, serial numbers that should not be public, computer screens containing private data, or third-party copyrighted graphics.

## Data and code availability

The current recommended statement for a manuscript is provided in [`DATA_CODE_AVAILABILITY.md`](DATA_CODE_AVAILABILITY.md).

For a permanent scholarly citation, archive a tagged GitHub release in a DOI-issuing repository such as Zenodo and then update both the manuscript and `CITATION.cff` with the DOI.

## Citation

If you use this repository, please cite the associated article and/or the archived software/data release.

GitHub can expose citation metadata from the root-level `CITATION.cff` file. Complete the author, repository URL, release version, and DOI fields before the final public release.

## Versioning

For the version supplied with a manuscript submission, create a fixed release, for example:

```text
v1.0.0
```

Use later releases for corrections or extensions rather than silently replacing files associated with the submitted manuscript.

## License

Licensing should be explicitly chosen before release. See [`LICENSES.md`](LICENSES.md) for a practical separation between source-code and research-data licensing.

## Contact

For questions about this repository, use the GitHub Issues page or the corresponding author contact information given in the associated manuscript.
