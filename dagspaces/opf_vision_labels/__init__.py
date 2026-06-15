"""opf_vision_labels — pseudo-label generation for the OPF vision head.

Runs off-the-shelf detectors (pedestrian, vehicle, blur-region) + a vision
LLM for text spotting over Cyclomedia panoramic faces, and emits a unified
parquet of per-detection rows (face / license_plate / house_number / person)
that feeds the OPF vision-head Stage 1 trainer in the sibling
`privacy-filter` repository.
"""
