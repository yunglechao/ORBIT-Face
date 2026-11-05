# Experiment figure sources

These assets visualize existing paper results. Generating them does not train models, extract face features, or rerun evaluations.

## Files

| Asset | Source and scope |
|---|---|
| `figures/tsne_alignment.png` | Rendered from the camera-ready `tsne_alignment.pdf`; no points or plot content were edited. The original caption identifies R100 circles, ViT-B triangles, and identity colors. This is a qualitative projection, not a retrieval metric. |
| `figures/ifrt_identification.svg` / `.png` | Four representative rows of the paper's original derived IFRT-MR identification table. All seven rows are included in the README table. |
| `figures/calibration_transfer.svg` / `.png` | Controlled cross-corpus evaluation comparing external LFW and identity-disjoint IFRT calibration, with old-old, direct, and full-backfill references. |
| `data/experiment_results.json` | Aggregate metrics, model names, protocol metadata, source filenames and SHA-256 digests, and record-level source mappings. Accuracy values are stored as fractions and displayed as percentages. |

No dataset images, identity labels, embeddings, or pretrained model weights are included in these assets. Local machine and server paths are omitted from the exported aggregate data.

## Regenerate the charts and tables

Run from the repository root:

```bash
python -m pip install numpy matplotlib
python tools/plot_results.py
```

The script regenerates the two quantitative charts as SVG and PNG files and updates the marked table blocks in `README.md`. It does not regenerate the t-SNE projection: that PNG is an existing paper figure rendered from PDF. Source-file digests are recorded in the aggregate JSON so the original artifacts can be checked when available.

## Original large-gallery table

- Dataset: the paper's original **derived IFRT-MR 1:N identification split**, with 10,000 probes and 1,614,305 gallery images. It is not the official IFRT/MFR-Ongoing verification protocol.
- Models: WebFace12M encoders; the exported rows retain exact old/new model identifiers and seeds.
- Old-old and direct values come from the `direct` fields of `ifrt_cross_ident_v2.json`. Those fields do not depend on calibration. The file also contains aligned results with `K=2,000`, which are **not used** here.
- Procrustes and Procrustes + MLP values come from `ifrt_proc_mlp.json`, with `K=5,000`.
- Every displayed value was checked against the camera-ready table. These are reported fixed-pair results, not a new multi-run estimate; no confidence intervals are inferred.
- The manuscript describes in-domain calibration, but the archived JSONs lack calibration/probe indices, hashes, and exclusion manifests. The exports therefore cannot independently establish calibration identity exclusion or exact split equivalence across the original runs. This limitation does not apply to the matched controlled-calibration comparison below, whose split hashes are available.
- Old-old is a legacy reference. It must not be interpreted as the new-model full-backfill result.

## Controlled calibration comparison

- Upgrade: `MS1MV2-R100-ArcFace` → `WebFace12M-R100-ArcFace-seed1`.
- Both runs use `K=3,000`, 10,000 probes, and the same 1,594,196-image gallery, with single-view features and no flip augmentation.
- Sources: `ifrt_fourway_ifrtcal_k3000_strictsplit.json` and `ifrt_fourway_lfwcal_k3000_strictsplit.json`.
- Probe-index and reserved-identity hashes match across the two files. Old-old, direct, and new-new Rank-1/Rank-5 also match exactly.
- All 3,000 reserved IFRT calibration identities, comprising 20,109 images, are excluded from the gallery. Calibration identities are disjoint from the probes. Identity labels are used to construct the split; the Procrustes fit uses paired embeddings only.
- External LFW calibration is applied to the same evaluation split. The comparison changes the calibration source, not the gallery or probe set.
- These results were added in the rebuttal/camera-ready materials and use a different split from the original large-gallery table.

The cross-corpus checkpoints also differ in training pipeline. These results do not establish that their training identities are disjoint, or that alignment recovers the full benefit of the newer model.
