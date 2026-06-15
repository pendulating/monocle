"""SCRFD face detection stage (P1, not used on Cyclomedia).

Kept for non-pre-blurred image sources and as a validator for the
blur-region + classify_blur pipeline. On Cyclomedia this stage will
find nothing because faces are already destroyed by the existing blur.

Default backbone: SCRFD via the ``insightface`` Python package; weights
fetched from HuggingFace (``public-data/insightface`` or pinned mirror).
"""

from __future__ import annotations

from dagspaces.common.orchestrator import StageExecutionContext, StageResult
from dagspaces.common.runners.base import StageRunner


class FaceDetectorRunner(StageRunner):
    stage_name = "face"

    def run(self, context: StageExecutionContext) -> StageResult:
        raise NotImplementedError(
            "FaceDetectorRunner.run is not yet implemented. P1 priority. "
            "Load SCRFD via insightface, run over input manifest, emit "
            "class='face'. Do not add to Cyclomedia pipelines."
        )
