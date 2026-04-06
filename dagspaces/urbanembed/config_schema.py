"""Config schema for urbanembed — re-exported from common."""
from dagspaces.common.config_schema import *  # noqa: F401, F403
from dagspaces.common.config_schema import (
    ArtifactSpec, SourceSpec, OutputSpec, PipelineNodeSpec,
    PipelineGraphSpec, load_pipeline_graph, resolve_output_root, iter_topologically,
)
