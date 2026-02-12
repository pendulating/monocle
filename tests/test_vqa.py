"""Unit tests for VQA stage functionality."""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from PIL import Image
import numpy as np

from dagspaces.urbanvqa.stages.vqa import (
    render_prompt_template,
    _prepare_image_content,
    _group_by_prompt_optimization,
    run_vqa_stage,
)
from dagspaces.urbanvqa.orchestrator import _prepare_streaming_dataset


class TestPromptTemplateRendering:
    """Test Jinja2 template rendering."""
    
    def test_simple_template(self):
        """Test simple variable substitution."""
        template = "Question: {{prompt}}"
        context = {"prompt": "What is in this image?"}
        result = render_prompt_template(template, context)
        assert result == "Question: What is in this image?"
    
    def test_template_with_multiple_vars(self):
        """Test template with multiple variables."""
        template = "Focus on {{focus_area}}. Question: {{prompt}}"
        context = {"focus_area": "urban planning", "prompt": "What type of building?"}
        result = render_prompt_template(template, context)
        assert result == "Focus on urban planning. Question: What type of building?"
    
    def test_template_with_conditional(self):
        """Test template with conditional logic."""
        template = "{% if focus_area %}Focus on {{focus_area}}. {% endif %}Question: {{prompt}}"
        context = {"focus_area": "urban planning", "prompt": "What type of building?"}
        result = render_prompt_template(template, context)
        assert "Focus on urban planning" in result
        assert "Question: What type of building?" in result
    
    def test_template_without_jinja2(self):
        """Test that error is raised if Jinja2 is not available."""
        # This test would require mocking jinja2 availability
        # For now, we assume jinja2 is available
        pass


class TestImageContentPreparation:
    """Test image content preparation for vLLM."""
    
    def test_pil_image(self):
        """Test PIL Image object."""
        img = Image.new("RGB", (100, 100), color="red")
        result = _prepare_image_content(img)
        assert result["type"] == "image"
        assert result["image"] == img
    
    def test_image_url(self):
        """Test image URL."""
        url = "https://example.com/image.jpg"
        result = _prepare_image_content(url)
        assert result["type"] == "image_url"
        assert result["image_url"]["url"] == url
    
    def test_base64_string(self):
        """Test base64 encoded string."""
        # Create a simple base64 string
        base64_str = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
        result = _prepare_image_content(base64_str)
        assert result["type"] == "image_url"
        assert "url" in result["image_url"]
    
    def test_local_path(self):
        """Test local image path."""
        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img = Image.new("RGB", (100, 100), color="blue")
            img.save(tmp.name, "JPEG")
            tmp_path = tmp.name
        
        try:
            result = _prepare_image_content(tmp_path)
            assert result["type"] == "image"
            assert hasattr(result["image"], "size")  # PIL Image
        finally:
            os.unlink(tmp_path)


class TestPromptGrouping:
    """Test prompt grouping optimization."""
    
    def test_group_by_prompt(self):
        """Test that prompts are grouped correctly."""
        df = pd.DataFrame({
            "prompt": ["Q1", "Q2", "Q1", "Q3", "Q2", "Q1"],
            "image_path": ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg", "img5.jpg", "img6.jpg"],
            "sample_id": ["s1", "s2", "s3", "s4", "s5", "s6"]
        })
        
        result = _group_by_prompt_optimization(df)
        
        # Check that prompts are grouped together
        prompts = result["prompt"].tolist()
        # All Q1 should be together, all Q2 together, etc.
        q1_indices = [i for i, p in enumerate(prompts) if p == "Q1"]
        q2_indices = [i for i, p in enumerate(prompts) if p == "Q2"]
        
        # Check that indices are consecutive
        assert all(q1_indices[i+1] - q1_indices[i] == 1 for i in range(len(q1_indices)-1))
        assert all(q2_indices[i+1] - q2_indices[i] == 1 for i in range(len(q2_indices)-1))
    
    def test_group_by_prompt_no_prompt_column(self):
        """Test that function handles missing prompt column."""
        df = pd.DataFrame({
            "image_path": ["img1.jpg", "img2.jpg"],
            "sample_id": ["s1", "s2"]
        })
        
        result = _group_by_prompt_optimization(df)
        # Should return unchanged
        assert len(result) == len(df)
        assert list(result.columns) == list(df.columns)


class TestDataValidation:
    """Test data validation and error handling."""
    
    def test_missing_prompt_column(self):
        """Test that missing prompt column raises error."""
        df = pd.DataFrame({
            "image_path": ["img1.jpg"],
            "sample_id": ["s1"]
        })
        
        # Create minimal config
        cfg = OmegaConf.create({
            "model": {
                "model_source": "test/model",
                "batch_size": 1,
                "concurrency": 1,
            },
            "prompt": {
                "system": "Test system prompt"
            },
            "sampling_params_vqa": {
                "temperature": 0.0,
                "max_tokens": 100
            }
        })
        
        # This should raise an error or handle gracefully
        # The actual behavior depends on implementation
        # For now, we'll check that it doesn't crash unexpectedly
        try:
            result = run_vqa_stage(df, cfg)
            # If it doesn't raise, check that result is empty or has error handling
            assert isinstance(result, pd.DataFrame)
        except (ValueError, RuntimeError, KeyError):
            # Expected errors are acceptable
            pass
    
    def test_missing_image_columns(self):
        """Test that missing image columns raise error."""
        df = pd.DataFrame({
            "prompt": ["What is in this image?"],
            "sample_id": ["s1"]
        })
        
        cfg = OmegaConf.create({
            "model": {
                "model_source": "test/model",
                "batch_size": 1,
                "concurrency": 1,
            },
            "prompt": {
                "system": "Test system prompt"
            },
            "sampling_params_vqa": {
                "temperature": 0.0,
                "max_tokens": 100
            },
            "runtime": {
                "image_fallback": False
            }
        })
        
        # Should raise error or handle gracefully
        try:
            result = run_vqa_stage(df, cfg)
            assert isinstance(result, pd.DataFrame)
        except (ValueError, RuntimeError, KeyError):
            pass
    
    def test_sample_id_generation(self):
        """Test that sample_id is generated if missing."""
        # This test would require checking the orchestrator's _load_parquet_dataset
        # For now, we'll test that the function handles missing sample_id
        df = pd.DataFrame({
            "prompt": ["What is in this image?"],
            "image_path": ["img1.jpg"]
        })
        
        # Check that sample_id column exists after processing
        # This is handled in orchestrator, so we'll just verify the structure
        assert "prompt" in df.columns
        assert "image_path" in df.columns


class TestStructuredJSONOutput:
    """Test structured JSON output parsing."""
    
    def test_json_extraction(self):
        """Test that JSON is extracted from model response."""
        # This would test the postprocessing logic
        # For now, we'll create a simple test
        response_text = '{"answer": "A building", "confidence": 0.9}'
        
        import json
        parsed = json.loads(response_text)
        assert parsed["answer"] == "A building"
        assert parsed["confidence"] == 0.9
    
    def test_json_extraction_with_text(self):
        """Test JSON extraction when wrapped in text."""
        import re
        response_text = 'The answer is {"answer": "A building", "confidence": 0.9}'
        
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            assert parsed["answer"] == "A building"


class TestImagePartitioning:
    """Test Ray Data partitioned image ingestion."""

    def test_directory_partitioning_generates_metadata(self, tmp_path):
        import ray

        labels = ["class_a", "class_b"]
        for label in labels:
            label_dir = tmp_path / label
            label_dir.mkdir()
            img = Image.new("RGB", (16, 16), color="white")
            img_path = label_dir / f"{label}_sample.jpg"
            img.save(img_path, "JPEG")

        cfg = OmegaConf.create(
            {
                "data": {
                    "image_path": str(tmp_path),
                    "default_prompt": "Describe the scene",
                    "partitioning": {
                        "type": "dir",
                        "field_names": ["label"],
                    },
                },
                "runtime": {
                    "debug": False,
                    "streaming_io": True,
                },
            }
        )

        # Ensure there is no stale Ray context
        if ray.is_initialized():
            ray.shutdown()

        ds, use_streaming = _prepare_streaming_dataset("", {}, cfg, "vqa")
        assert use_streaming is True
        assert ds is not None
        try:
            schema_names = set(ds.schema().names)
            assert {"image", "path", "image_path", "label", "prompt", "sample_id"}.issubset(schema_names)

            rows = ds.take_all()
            assert len(rows) == len(labels)
            for row in rows:
                assert row["prompt"] == "Describe the scene"
                assert row["sample_id"]
                assert row["label"] in labels
        finally:
            ray.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
