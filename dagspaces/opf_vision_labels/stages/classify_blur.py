"""Classify blur-candidate regions as face or license_plate.

Joins the outputs of ``blur_region``, ``person``, and ``vehicle`` on
``sample_id`` and assigns each blur candidate a final class via IoU /
containment heuristics:

  - Candidate centroid inside a person's upper 25-30% → ``face``.
  - Candidate IoU with any vehicle bbox above threshold → ``license_plate``.
  - Neither → dropped (ambiguous).

Emits per-detection rows with the final class, preserving the original
blur-candidate bbox and score.
"""

from __future__ import annotations

from dagspaces.common.orchestrator import StageExecutionContext, StageResult
from dagspaces.common.runners.base import StageRunner


class ClassifyBlurRunner(StageRunner):
    stage_name = "classify_blur"

    def run(self, context: StageExecutionContext) -> StageResult:
        raise NotImplementedError(
            "ClassifyBlurRunner.run is not yet implemented. "
            "Read blur_region + person + vehicle parquets from "
            "context.inputs; for each blur candidate compute IoU vs. "
            "person-upper-body and vehicle bboxes; assign "
            "class='face' | 'license_plate' or drop."
        )
