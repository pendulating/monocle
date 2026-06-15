"""YOLOv8 license-plate detection stage (P1, not used on Cyclomedia).

Kept for non-pre-blurred image sources. On Cyclomedia this stage will
find nothing because plates are already destroyed by the existing blur.

Default backbone: YOLOv8-plate (``morsetechlab/yolov11-license-plate-
detection`` on HuggingFace, loaded via ``ultralytics``). Note: the
weights are AGPLv3 — review redistribution implications before
publishing any model trained with them as the teacher.
"""

from __future__ import annotations

from dagspaces.common.orchestrator import StageExecutionContext, StageResult
from dagspaces.common.runners.base import StageRunner


class PlateDetectorRunner(StageRunner):
    stage_name = "plate"

    def run(self, context: StageExecutionContext) -> StageResult:
        raise NotImplementedError(
            "PlateDetectorRunner.run is not yet implemented. P1 priority. "
            "Load YOLOv8-plate via ultralytics, run over input manifest, "
            "emit class='license_plate'. Do not add to Cyclomedia pipelines."
        )
