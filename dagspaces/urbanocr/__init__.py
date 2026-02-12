"""Urban OCR Dagspace - Text spotting in street view imagery.

This dagspace provides OCR/text spotting capabilities using vision-language models
(Qwen3-VL) with the same vLLM + Ray Data + Hydra pipeline infrastructure as urbanvqa.

Key features:
- Generic OCR processing with pluggable data handlers
- Full text localization with bounding boxes and confidence scores
- Cyclomedia street view support via dedicated data handler
- Flat parquet output (one row per text detection)
"""

