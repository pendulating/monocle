"""The extraction schemas in research/conf/pairwisevqa/extract/ load and align."""

import pytest
from omegaconf import OmegaConf

from dagspaces.common.langextract_backend import (
    validate_examples,
    spec_from_config,
)


class TestTheShippedSchemas:
    """The config groups of the paper corpus (v1 to v3) and of F4 (single-v1)."""

    # The repo root, thus the tests also pass when pytest starts in another folder.
    CONF = str(__import__("pathlib").Path(__file__).resolve().parent.parent / "research/conf/pairwisevqa/extract")

    @pytest.mark.parametrize("version", ["v1", "v2"])
    def test_the_examples_align(self, version):
        spec = spec_from_config(OmegaConf.load(f"{self.CONF}/urban_cues_{version}.yaml"))
        assert validate_examples(spec) == []
        assert len(spec.classes) == 7

    def test_v2_holds_the_longest_qwen_trace(self):
        """The qwen3.5-9b runs of 2026-08-14 reach 35,598 characters."""
        spec = spec_from_config(OmegaConf.load(f"{self.CONF}/urban_cues_v2.yaml"))
        assert spec.max_char_buffer > 35598

    @pytest.mark.parametrize("name,version", [
        ("urban_cues_v2", "v2"),
        ("urban_cues_v3", "v3"),
        ("urban_cues_single_v1", "single-v1"),
    ])
    def test_enums_sit_inside_the_analysis_vocabulary(self, name, version):
        """`_extractions.VOCABULARY` is the union of the values of every
        version, thus each value a version declares must be in it. A value
        that is absent there becomes `other` in the analysis."""

        from research.notebooks.cvpr import _extractions as X  # noqa: E402

        spec = spec_from_config(OmegaConf.load(f"{self.CONF}/{name}.yaml"))
        assert spec.schema_version == version
        assert version in X.SCHEMA_VERSIONS
        for klass, attrs in spec.attribute_values.items():
            assert klass in X.VOCABULARY, klass
            for attr, values in attrs.items():
                have = {str(v) for v in X.VOCABULARY[klass][attr]}
                missing = [v for v in values if v not in have]
                assert not missing, (klass, attr, missing)
        assert set(spec.classes) == set(X.CLASSES_BY_VERSION[version])

    def test_single_v1_examples_align_and_hold_the_longest_f4_text(self):
        """The longest F4 text is a qwen3.5-9b trace of 55,231 characters."""
        spec = spec_from_config(OmegaConf.load(f"{self.CONF}/urban_cues_single_v1.yaml"))
        assert validate_examples(spec) == []
        assert len(spec.classes) == 5
        assert spec.max_char_buffer > 55231

    def test_v3_examples_align_and_teach_every_class(self):
        spec = spec_from_config(OmegaConf.load(f"{self.CONF}/urban_cues_v3.yaml"))
        assert validate_examples(spec) == []
        assert len(spec.classes) == 12
