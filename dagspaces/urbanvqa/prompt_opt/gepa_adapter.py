from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from .lm_resolver import resolve_lm_clients

# Persistent processor for model reuse across iterations
_PERSISTENT_PROCESSOR = None

try:
    from gepa.core.adapter import GEPAAdapter, EvaluationBatch
except ImportError:  # pragma: no cover - gepa is optional during local dev
    class GEPAAdapter:  # type: ignore[too-many-ancestors]
        """Fallback shim so type-checkers see the interface."""

        def evaluate(self, batch: Sequence[Any], candidate: Mapping[str, Any], capture_traces: bool = False) -> Any:
            raise NotImplementedError

        def make_reflective_dataset(self, candidate: Mapping[str, Any], eval_batch: Any, components_to_update: List[str]) -> Mapping[str, Sequence[Mapping[str, Any]]]:
            raise NotImplementedError


LOG = logging.getLogger(__name__)


def _normalize_mapping_batch(batch: Mapping[str, Any]) -> pd.DataFrame:
    if not batch:
        return pd.DataFrame()
    # If values are sequences of equal length, construct dataframe directly.
    if all(
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        for value in batch.values()
    ):
        lengths = {len(value) for value in batch.values()}
        if len(lengths) == 1:
            return pd.DataFrame(batch)
    # Fallback: treat mapping as a single record.
    return pd.DataFrame([batch])


def _coerce_to_dataframe(batch: Sequence[Any] | Mapping[str, Any]) -> pd.DataFrame:
    if isinstance(batch, pd.DataFrame):
        return batch.copy()
    if isinstance(batch, Mapping):
        return _normalize_mapping_batch(batch)
    if not isinstance(batch, Sequence):
        raise TypeError(f"Unsupported minibatch type: {type(batch)}")
    if len(batch) == 0:
        return pd.DataFrame()
    first = batch[0]
    if isinstance(first, pd.DataFrame):
        return pd.concat(batch, ignore_index=True)
    if isinstance(first, Mapping):
        return pd.DataFrame(list(batch))
    raise TypeError(f"Unsupported minibatch element type: {type(first)}")


class GEPAVQAAdapter(GEPAAdapter):
    """Adapter that evaluates GEPA candidates with the VQA stage stack.
    
    This adapter supports two modes:
    1. Standard mode: Creates a new vLLM processor for each evaluation (default)
    2. Persistent mode: Reuses a single vLLM processor across all evaluations
       (enabled via gepa.optimization.persistent_processor: true)
    
    Persistent mode eliminates model teardown/reload overhead between GEPA
    iterations, significantly speeding up optimization.
    """

    def __init__(
        self,
        base_cfg: DictConfig,
    ) -> None:
        self._base_cfg = deepcopy(base_cfg)
        metrics_root = getattr(base_cfg.gepa, "metrics", OmegaConf.create({}))
        metrics_dict = OmegaConf.to_container(metrics_root, resolve=True) or {}
        self._primary_metric_name = metrics_dict.get("primary", "accuracy")
        secondary_cfg = metrics_dict.get("secondary", [])
        if isinstance(secondary_cfg, str):
            secondary_cfg = [secondary_cfg]
        self._metric_specs: Dict[str, Dict[str, Any]] = {}
        for name, spec in metrics_dict.items():
            if name in {"primary", "secondary"}:
                continue
            if isinstance(spec, dict):
                self._metric_specs[name] = spec
        if self._primary_metric_name not in self._metric_specs:
            raise ValueError(f"Primary metric '{self._primary_metric_name}' is not defined in cfg.gepa.metrics")
        self._metric_order: List[str] = [self._primary_metric_name]
        for metric_name in secondary_cfg:
            if metric_name in self._metric_specs and metric_name not in self._metric_order:
                self._metric_order.append(metric_name)
        # Append remaining metrics for logging completeness
        for metric_name in self._metric_specs:
            if metric_name not in self._metric_order:
                self._metric_order.append(metric_name)

        primary_spec = self._metric_specs[self._primary_metric_name]
        self._reference_column = primary_spec.get("reference_column", "expected_answer")
        self._prediction_column = primary_spec.get("prediction_column", "answer")
        
        # Build component name -> target path mapping from gepa.components config
        # This allows dynamic configuration of which prompts to optimize
        self._component_targets: Dict[str, str] = {}
        components_cfg = getattr(base_cfg.gepa, "components", None)
        if components_cfg:
            for comp in components_cfg:
                name = comp.get("name")
                target = comp.get("target")
                if name and target:
                    self._component_targets[name] = target
        else:
            # Legacy fallback: use hardcoded paths
            self._component_targets = {
                "system_prompt": "prompt.system",
                "user_prompt": "prompt.template",
            }
        
        self._lm_clients = resolve_lm_clients(base_cfg)
        
        # Check if persistent processor mode is enabled
        # NOTE: We do NOT initialize the processor here! Initialization must happen
        # lazily in evaluate() to avoid serialization issues when submitting to SLURM.
        optimization_cfg = getattr(base_cfg.gepa, "optimization", OmegaConf.create({}))
        self._use_persistent_processor = getattr(optimization_cfg, "persistent_processor", False)
        self._persistent_processor = None
        self._persistent_processor_init_attempted = False
        
        # Counter for full validation rounds (used for FN logging)
        # Increments only when we log false negatives on full validation
        self._validation_round = 0

    def _ensure_persistent_processor_initialized(self) -> bool:
        """Lazily initialize the persistent vLLM or Cambrian processor for model reuse.
        
        This method is called on the first evaluate() call, NOT in __init__.
        The initialization must happen inside the worker process.
        
        Returns:
            True if processor is ready, False if fallback to standard mode is needed.
        """
        global _PERSISTENT_PROCESSOR
        
        # Already initialized successfully
        if self._persistent_processor is not None:
            return True
        
        # Already tried and failed
        if self._persistent_processor_init_attempted:
            return False
        
        self._persistent_processor_init_attempted = True
        
        # Check if there's a global processor we can reuse
        if _PERSISTENT_PROCESSOR is not None:
            LOG.info("Reusing existing persistent processor")
            self._persistent_processor = _PERSISTENT_PROCESSOR
            return True
        
        # Detect if model is Cambrian based on model_source
        model_source = str(getattr(self._base_cfg.model, "model_source", "")).lower()
        is_cambrian = "cambrian" in model_source
        
        try:
            if is_cambrian:
                from dagspaces.urbanvqa.stages.persistent_cambrian import PersistentCambrianProcessor
                LOG.info("Initializing persistent Cambrian processor for GEPA optimization (lazy init)")
                self._persistent_processor = PersistentCambrianProcessor.get_or_create(self._base_cfg)
            else:
                from dagspaces.urbanvqa.stages.persistent_vllm import PersistentVLLMProcessor
                LOG.info("Initializing persistent vLLM processor for GEPA optimization (lazy init)")
                self._persistent_processor = PersistentVLLMProcessor.get_or_create(self._base_cfg)
            
            self._persistent_processor.initialize()
            _PERSISTENT_PROCESSOR = self._persistent_processor
            LOG.info("Persistent processor initialized - model will be reused across iterations")
            return True
            
        except Exception as e:
            LOG.warning(f"Failed to initialize persistent processor: {e}. Falling back to standard mode.")
            import traceback
            LOG.debug(traceback.format_exc())
            self._use_persistent_processor = False
            self._persistent_processor = None
            return False

    def evaluate(
        self,
        batch: Sequence[Any],
        candidate: Mapping[str, Any],
        *,
        capture_traces: bool = True,
    ) -> EvaluationBatch[Dict[str, Any], Dict[str, Any]]:
        batch_df = _coerce_to_dataframe(batch)
        
        if batch_df.empty:
            return EvaluationBatch(outputs=[], scores=[], trajectories=[])

        cfg = self._configure_candidate(candidate)
        
        # Use persistent processor if enabled (with lazy initialization), otherwise standard VQA stage
        results: pd.DataFrame
        if self._use_persistent_processor:
            ready = self._ensure_persistent_processor_initialized()
            if ready and self._persistent_processor:
                results = self._evaluate_with_persistent_processor(batch_df, candidate, cfg)
            else:
                # If persistent initialization failed for Cambrian, DO NOT fallback to standard VQA stage
                # as it is guaranteed to fail with Transformers architecture error in the main process.
                model_source = str(getattr(self._base_cfg.model, "model_source", "")).lower()
                if "cambrian" in model_source:
                    LOG.error("Persistent Cambrian processor failed to initialize. Aborting to avoid standard fallback error.")
                    raise RuntimeError("Failed to initialize PersistentCambrianProcessor. Check logs for actor initialization errors.")
                
                # Fallback for non-Cambrian models
                from dagspaces.urbanvqa.stages.vqa import run_vqa_stage
                return run_vqa_stage(batch_df, cfg)
        else:
            # Standard mode (reloads model every time)
            from dagspaces.urbanvqa.stages.vqa import run_vqa_stage
            # run_vqa_stage returns an EvaluationBatch, so we can return directly here
            # or we could adapt it to return a DataFrame for consistency.
            # For now, let's keep the standard path as is.
            return run_vqa_stage(batch_df, cfg)
        
        # Shared processing logic for persistent processor results
        merged = self._merge_expected(results, batch_df)
        
        # Compute per-sample scores
        scores = self._compute_sample_scores(merged)
        
        # Compute and log all metrics (primary + secondary) for full validation batches
        # Full validation batches are typically larger than subsample batches (usually >100 samples)
        if len(batch_df) >= 100:
            self._log_all_metrics(merged)
            
            # Log false negatives for debugging (after each full validation)
            debug_cfg = getattr(self._base_cfg.gepa, "debug", OmegaConf.create({}))
            if getattr(debug_cfg, "log_false_negatives", True):
                self._log_false_negatives(
                    merged,
                    batch_df,
                    iteration=self._validation_round,
                    max_images_to_log=getattr(debug_cfg, "max_fn_images", 20),
                )
                self._validation_round += 1  # Increment AFTER logging to stay aligned
        
        # Build traces/trajectories
        traces = self._build_traces(merged, candidate)
        
        # Build outputs (using traces as output representation for now, or could be just model responses)
        outputs = traces
        
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=traces if capture_traces else None
        )
    
    def _evaluate_with_persistent_processor(
        self,
        batch_df: pd.DataFrame,
        candidate: Mapping[str, Any],
        cfg: DictConfig,
    ) -> pd.DataFrame:
        """Evaluate using the persistent processor with current candidate prompts."""
        # Extract prompts from candidate
        system_prompt = candidate.get("system_prompt", "")
        user_template = candidate.get("user_prompt", "")
        
        # Use the persistent processor
        return self._persistent_processor.evaluate(
            df=batch_df,
            system_prompt=system_prompt,
            user_template=user_template,
            cfg=cfg,
        )

    def make_reflective_dataset(
        self,
        candidate: Dict[str, str],
        eval_batch: EvaluationBatch[Dict[str, Any], Dict[str, Any]],
        components_to_update: List[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Build a reflective dataset for the teacher LLM.
        
        When multimodal reflection is enabled, this method embeds image markers
        in the records so the reflection VLM can see the actual images.
        Images are selected with a bias toward incorrect predictions (default 70%)
        to help the reflector focus on failure cases.
        """
        dataset: Dict[str, List[Mapping[str, Any]]] = {}
        if not eval_batch.trajectories:
            return {}

        # Check if multimodal reflection is enabled
        optimization_cfg = getattr(self._base_cfg.gepa, "optimization", OmegaConf.create({}))
        use_multimodal = getattr(optimization_cfg, "multimodal_reflection", False)
        
        # Multimodal reflection settings
        max_images = getattr(optimization_cfg, "reflection_max_images", 5)
        incorrect_ratio = getattr(optimization_cfg, "reflection_incorrect_ratio", 0.7)
        
        # Determine which traces should include images (if multimodal enabled)
        traces_with_images: set = set()
        if use_multimodal:
            from .multimodal_reflection import select_images_for_reflection
            traces_with_scores = list(zip(eval_batch.trajectories, eval_batch.scores))
            selected_traces = select_images_for_reflection(
                traces_with_scores,
                max_images=max_images,
                incorrect_ratio=incorrect_ratio,
            )
            # Use sample_id to track which traces get images
            traces_with_images = {t.get("sample_id") for t in selected_traces if t.get("sample_id")}

        for component in components_to_update:
            records = []
            for trace, score in zip(eval_batch.trajectories, eval_batch.scores):
                prompt = trace.get("prompt", "")
                answer = trace.get("answer", "")
                expected = trace.get("expected_answer", "")
                sample_id = trace.get("sample_id")
                
                # Format feedback for the teacher
                feedback = f"Expected: {expected}. Predicted: {answer}. Score: {score}"
                
                # Build inputs dict
                inputs: Dict[str, Any] = {"user_prompt": prompt}
                
                # Include image marker if this trace was selected for multimodal
                if use_multimodal and sample_id in traces_with_images:
                    image_ref = trace.get("image_ref")
                    if image_ref:
                        image_marker = self._format_image_marker(image_ref)
                        inputs["image"] = image_marker
                
                record = {
                    "Inputs": inputs,
                    "Generated Outputs": answer,
                    "Feedback": feedback,
                    "score": score
                }
                records.append(record)
            dataset[component] = records
            
        return dataset
    
    def _format_image_marker(self, image_ref: str) -> str:
        """Format image reference as a parseable marker for multimodal reflection.
        
        The marker format [IMAGE:...] is parsed by the multimodal reflection LM
        wrapper and converted to proper multimodal message format.
        
        Args:
            image_ref: Image path, URL, or base64 data
            
        Returns:
            Formatted marker string like [IMAGE:file:///path/to/image.jpg]
        """
        if not image_ref:
            return ""
        
        if image_ref.startswith("data:image"):
            # Already base64 data URL
            return f"[IMAGE:{image_ref}]"
        elif image_ref.startswith("http://") or image_ref.startswith("https://"):
            # Remote URL
            return f"[IMAGE:{image_ref}]"
        else:
            # Local file path - use file:// URI scheme
            return f"[IMAGE:file://{image_ref}]"

    @staticmethod
    def _extract_binary_answer(text: str, positive_class: str = "yes", negative_class: str = "no") -> str:
        """Extract binary answer from model response.
        
        Models often provide explanations like "Yes, there is flooding because..."
        This function extracts just the binary answer by looking for yes/no patterns.
        
        Priority order:
        1. If response starts with positive/negative class word, use that
        2. If response contains positive/negative class word as a standalone word in the first 50 chars, use that
        3. Otherwise return the original text (for exact match fallback)
        
        Args:
            text: Model response text
            positive_class: Positive class label (default "yes")
            negative_class: Negative class label (default "no")
            
        Returns:
            Normalized binary answer or original text if no match found
        """
        if not text:
            return ""
        
        text_lower = text.lower().strip()
        positive_class = positive_class.lower()
        negative_class = negative_class.lower()
        
        # 1. Check if response starts with the answer (most reliable)
        # Handle cases like "Yes.", "Yes,", "Yes -", "Yes:"
        import re
        starts_positive = re.match(rf'^{re.escape(positive_class)}[\s.,:\-!]', text_lower) or text_lower == positive_class
        starts_negative = re.match(rf'^{re.escape(negative_class)}[\s.,:\-!]', text_lower) or text_lower == negative_class
        
        if starts_positive:
            return positive_class
        if starts_negative:
            return negative_class
        
        # 2. Look for standalone word in the beginning of the text (more lenient)
        # This catches "Based on the image, yes." or "I think the answer is no."
        # We only look at the first 100 characters to avoid catching "No" in a later explanation
        prefix = text_lower[:100]
        contains_positive = re.search(rf'\b{re.escape(positive_class)}\b', prefix)
        contains_negative = re.search(rf'\b{re.escape(negative_class)}\b', prefix)
        
        if contains_positive and not contains_negative:
            return positive_class
        if contains_negative and not contains_positive:
            return negative_class
            
        # 3. If both are found, prefer the one that appears first
        if contains_positive and contains_negative:
            if contains_positive.start() < contains_negative.start():
                return positive_class
            else:
                return negative_class
        
        # No match found - return original for exact match fallback
        return text_lower

    def _compute_sample_scores(self, frame: pd.DataFrame) -> List[float]:
        """Compute per-sample scores for the batch.
        
        For NPV/FOR optimization, uses asymmetric weighting to penalize
        False Negatives (missed floods) more heavily than False Positives.
        
        Note: Uses lenient answer extraction - if the model says "Yes, because..."
        it will be counted as "yes" rather than a mismatch.
        """
        primary_spec = self._metric_specs.get(self._primary_metric_name, {})
        prediction_column = primary_spec.get("prediction_column", self._prediction_column)
        reference_column = primary_spec.get("reference_column", self._reference_column)
        positive_class = primary_spec.get("positive_class", "yes").lower()
        negative_class = "no"  # Assumed binary classification
        
        # Extract binary answers from potentially longer responses
        raw_predictions = frame[prediction_column].fillna("").astype(str)
        predictions = raw_predictions.apply(
            lambda x: self._extract_binary_answer(x, positive_class, negative_class)
        )
        expected = frame[reference_column].fillna("").astype(str).str.strip().str.lower()
        
        # Check if we need asymmetric scoring for NPV/FOR optimization
        metric_key = self._primary_metric_name.lower()
        
        if metric_key in {"f1", "accuracy", "exact_match"}:
            # Standard symmetric scoring: 1.0 for match, 0.0 for mismatch
            # This is the default for F1 optimization in GEPA.
            return (predictions == expected).astype(float).tolist()

        elif metric_key in {"npv", "negative_predictive_value", "for", "false_omission_rate"}:
            # EXTREME asymmetric scoring for NPV/FOR optimization:
            # - Correct predictions: 1.0
            # - False Positives (pred=Yes, truth=No): 1.0 (NO penalty - false alarms are FREE)
            # - False Negatives (pred=No, truth=Yes): -5.0 (CATASTROPHIC penalty - missed flood is disaster)
            #
            # Gap of 6.0 points per FN ensures optimizer will do ANYTHING to avoid missing floods.
            # This effectively makes the optimizer maximize recall at all costs.
            scores = []
            for pred, truth in zip(predictions, expected):
                if pred == truth:
                    scores.append(1.0)  # Correct (TP or TN)
                elif pred != positive_class and truth == positive_class:
                    scores.append(0)  # False Negative - CATASTROPHIC (missed flood!)
                else:
                    scores.append(0.5)  # False Positive - NO penalty (false alarm is acceptable)
            return scores
        
        elif metric_key in {"precision", "ppv", "positive_predictive_value"}:
            # For precision optimization: penalize False Positives more
            # - Correct predictions: 1.0
            # - False Negatives: 0.5 (moderate penalty)
            # - False Positives: 0.0 (severe penalty - this is what precision penalizes)
            scores = []
            for pred, truth in zip(predictions, expected):
                if pred == truth:
                    scores.append(1.0)
                elif pred == positive_class and truth != positive_class:
                    scores.append(0.0)  # False Positive - severely penalized
                else:
                    scores.append(0.5)  # False Negative - moderately penalized
            return scores
        
        elif metric_key in {"recall", "sensitivity", "tpr"}:
            # For recall optimization: same extreme scoring as NPV/FOR
            scores = []
            for pred, truth in zip(predictions, expected):
                if pred == truth:
                    scores.append(1.0)
                elif pred != positive_class and truth == positive_class:
                    scores.append(0)  # False Negative - Penalized
                else:
                    scores.append(0.5)  # False Positive - Moderately penalized
            return scores
        
        else:
            # Default to exact match for accuracy, F1, etc.
            scores = (predictions == expected).astype(float).tolist()
            return scores
    
    def _log_all_metrics(self, frame: pd.DataFrame) -> None:
        """Compute and log all metrics (primary + secondary) for the evaluation.
        
        This logs metrics to both stdout and wandb (if available) to track
        precision, recall, false_omission_rate, etc. during optimization.
        """
        try:
            all_metrics = self._compute_metrics(frame)
            
            # Log to stdout
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in all_metrics.items())
            LOG.info(f"Evaluation metrics: {metrics_str}")
            print(f"[GEPA] All metrics: {metrics_str}", flush=True)
            
            # Log to wandb if available
            try:
                from dagspaces.common.wandb_logger import WandbLogger
                import wandb
                if wandb.run is not None:
                    # Prefix metrics with gepa/ for wandb
                    wandb_metrics = {f"gepa/metric_{k}": v for k, v in all_metrics.items()}
                    wandb.log(wandb_metrics, commit=False)
            except Exception as e:
                LOG.debug(f"Could not log metrics to wandb: {e}")
                
        except Exception as e:
            LOG.warning(f"Failed to compute/log all metrics: {e}")

    def _log_false_negatives(
        self,
        frame: pd.DataFrame,
        batch_df: pd.DataFrame,
        iteration: int,
        max_images_to_log: int = 20,
    ) -> None:
        """Log false negative samples for debugging after full validation.
        
        False negatives are samples where:
        - Model predicted negative (e.g., "No flood")
        - Ground truth is positive (e.g., actually flooded)
        
        These contribute to FOR (False Omission Rate) and are critical errors
        for flood detection where missing a flood is dangerous.
        
        Args:
            frame: DataFrame with predictions and ground truth
            batch_df: Original batch DataFrame (may have image paths)
            iteration: Current validation round number
            max_images_to_log: Max images to upload to WandB (bandwidth limit)
        """
        try:
            primary_spec = self._metric_specs.get(self._primary_metric_name, {})
            prediction_column = primary_spec.get("prediction_column", self._prediction_column)
            reference_column = primary_spec.get("reference_column", self._reference_column)
            positive_class = primary_spec.get("positive_class", "yes").lower()
            negative_class = "no"  # Assumed binary classification
            
            # Use lenient extraction for predictions (handle "Yes, because..." responses)
            raw_predictions = frame[prediction_column].fillna("").astype(str)
            predictions = raw_predictions.apply(
                lambda x: self._extract_binary_answer(x, positive_class, negative_class)
            )
            expected = frame[reference_column].fillna("").astype(str).str.strip().str.lower()
            
            # Identify false negatives: predicted negative, actually positive
            fn_mask = (predictions != positive_class) & (expected == positive_class)
            fn_frame = frame[fn_mask].copy()
            
            # Also get corresponding rows from batch_df for image paths
            if "sample_id" in fn_frame.columns and "sample_id" in batch_df.columns:
                fn_sample_ids = set(fn_frame["sample_id"].dropna().tolist())
                fn_batch_rows = batch_df[batch_df["sample_id"].isin(fn_sample_ids)]
            else:
                fn_batch_rows = batch_df.iloc[fn_mask.values] if len(batch_df) == len(frame) else pd.DataFrame()
            
            fn_count = len(fn_frame)
            total_negatives = (predictions != positive_class).sum()
            
            LOG.info(f"[Validation {iteration}] False negatives: {fn_count} / {total_negatives} negative predictions (FOR contributors)")
            print(f"[GEPA] Validation {iteration}: {fn_count} false negatives (missed floods)", flush=True)
            
            if fn_count == 0:
                return
            
            # Build records with image paths
            fn_records = []
            for i, (idx, row) in enumerate(fn_frame.iterrows()):
                # Try to get image path from batch_df
                image_path = None
                sample_id = row.get("sample_id")
                if sample_id is not None and not fn_batch_rows.empty:
                    batch_row = fn_batch_rows[fn_batch_rows["sample_id"] == sample_id]
                    if not batch_row.empty:
                        image_path = (
                            batch_row.iloc[0].get("image_path") or
                            batch_row.iloc[0].get("image_url") or
                            batch_row.iloc[0].get("image_base64")
                        )
                
                # Fallback to frame columns
                if not image_path:
                    image_path = row.get("image_path") or row.get("image_url") or ""
                
                record = {
                    "sample_id": str(sample_id) if sample_id else f"idx_{idx}",
                    "image_path": str(image_path) if image_path else "",
                    "prediction": str(row.get(prediction_column, "")),
                    "expected": str(row.get(reference_column, "")),
                    "prompt": str(row.get("prompt", ""))[:200],  # Truncate for readability
                }
                fn_records.append(record)
            
            # 1. Write to file
            self._write_fn_to_file(fn_records, iteration)
            
            # 2. Log to WandB
            self._log_fn_to_wandb(fn_records, iteration, max_images_to_log)
            
        except Exception as e:
            LOG.warning(f"Failed to log false negatives: {e}")
            import traceback
            LOG.debug(traceback.format_exc())

    def _write_fn_to_file(self, fn_records: List[Dict[str, Any]], iteration: int) -> None:
        """Write false negative records to a JSON file."""
        import json
        from pathlib import Path
        
        try:
            artifact_dir = Path(getattr(
                self._base_cfg.gepa.artifacts, "base_dir", "outputs/gepa"
            ))
            artifact_dir.mkdir(parents=True, exist_ok=True)
            
            fn_path = artifact_dir / f"false_negatives_val{iteration}.json"
            
            with open(fn_path, "w") as f:
                json.dump({
                    "validation_round": iteration,
                    "count": len(fn_records),
                    "samples": fn_records,
                }, f, indent=2)
            
            LOG.info(f"Wrote {len(fn_records)} false negatives to {fn_path}")
            
        except Exception as e:
            LOG.warning(f"Failed to write FN file: {e}")

    def _log_fn_to_wandb(
        self,
        fn_records: List[Dict[str, Any]],
        iteration: int,
        max_images: int,
    ) -> None:
        """Log false negatives to WandB with image visualization.
        
        Creates two artifacts:
        1. A table with embedded images (up to max_images) for visual inspection
        2. A full text table with all FN sample details
        """
        try:
            import wandb
            if wandb.run is None:
                return
            
            from pathlib import Path
            
            # Build all log data in a single dict for efficiency
            log_data: Dict[str, Any] = {
                "gepa/false_negative_count": len(fn_records),
                "gepa/validation_round": iteration,
            }
            
            # 1. Create a visual table with embedded images (limited to max_images)
            # This allows visual inspection of FN samples in WandB UI.
            #
            # IMPORTANT: Use a stable key (no iteration suffix) so WandB shows one
            # "living" table panel instead of cluttering the UI with many tables.
            visual_columns = ["validation_round", "image", "sample_id", "prediction", "expected"]
            visual_data = []
            images_loaded = 0
            
            for record in fn_records:
                if images_loaded >= max_images:
                    break
                
                image_path = record.get("image_path", "")
                if not image_path:
                    continue
                
                path = Path(image_path)
                if not path.exists():
                    continue
                
                try:
                    # Create wandb.Image with caption for the table
                    img = wandb.Image(
                        str(path),
                        caption=f"FN: pred={record['prediction']}, true={record['expected']}"
                    )
                    visual_data.append([
                        iteration,
                        img,
                        record["sample_id"],
                        record["prediction"],
                        record["expected"],
                    ])
                    images_loaded += 1
                except Exception as img_err:
                    LOG.debug(f"Could not load image {image_path}: {img_err}")
            
            if visual_data:
                visual_table = wandb.Table(columns=visual_columns, data=visual_data)
                log_data["gepa/fn_images"] = visual_table
                LOG.info(f"Created visual FN table with {len(visual_data)} images for validation {iteration}")
            
            # 2. Create a full text table with ALL FN samples (no images, just paths).
            # This is useful for programmatic analysis.
            #
            # IMPORTANT: Use a stable key (no iteration suffix) so WandB shows one
            # "living" table panel instead of cluttering the UI with many tables.
            text_columns = ["validation_round", "sample_id", "image_path", "prediction", "expected"]
            text_data = [
                [iteration, r["sample_id"], r["image_path"], r["prediction"], r["expected"]]
                for r in fn_records
            ]
            text_table = wandb.Table(columns=text_columns, data=text_data)
            log_data["gepa/fn_details"] = text_table
            
            # Log everything in a single call
            wandb.log(log_data, commit=False)
            LOG.info(f"Logged {len(fn_records)} FN records to WandB for validation {iteration}")
            
        except ImportError:
            pass  # wandb not available
        except Exception as e:
            LOG.warning(f"Failed to log FN to WandB: {e}")
            import traceback
            LOG.debug(traceback.format_exc())

    @property
    def lm_clients(self):
        return self._lm_clients

    def build_litellm_options(self) -> Dict[str, Dict[str, Any]]:
        """Return LiteLLM-compatible kwargs for reflection client.
        
        Note: llm.task is deprecated - VQA inference uses cfg.model directly.
        """
        return {name: client.as_litellm_kwargs() for name, client in self._lm_clients.items()}

    def _configure_candidate(self, candidate: Mapping[str, Any]) -> DictConfig:
        """Apply candidate prompts to config using component targets from gepa.components."""
        cfg = deepcopy(self._base_cfg)
        for component_name, target_path in self._component_targets.items():
            if component_name in candidate:
                OmegaConf.update(cfg, target_path, candidate[component_name], merge=True)
        return cfg

    def _merge_expected(self, results: pd.DataFrame, batch_df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the result frame has the reference column required for metrics."""
        if self._reference_column in results.columns:
            return results

        # Gather candidate columns from the minibatch for fallbacks.
        candidate_columns: List[str] = []
        seen: set[str] = set()

        def _maybe_add(name: Optional[str]) -> None:
            if name and name in batch_df.columns and name not in seen:
                candidate_columns.append(name)
                seen.add(name)

        # Priority order: configured reference column, dataset-provided alias, common aliases.
        _maybe_add(self._reference_column)
        dataset_expected = OmegaConf.select(self._base_cfg, "data.columns.expected_answer")
        _maybe_add(dataset_expected)
        for alias in ("expected_answer", "gt", "label", "answer", "target"):
            _maybe_add(alias)

        merged: Optional[pd.DataFrame] = None
        for column in candidate_columns:
            if "sample_id" in results.columns and "sample_id" in batch_df.columns:
                candidate = results.merge(
                    batch_df[[column, "sample_id"]],
                    on="sample_id",
                    how="left",
                    suffixes=("", "_expected"),
                )
                normalized = column if column == self._reference_column else f"{column}_expected"
                if normalized in candidate.columns:
                    candidate[self._reference_column] = candidate[normalized]
                    if normalized != self._reference_column:
                        candidate.drop(columns=[normalized], inplace=True)
                    merged = candidate
                    break
            else:
                # Fall back to positional alignment when sample IDs are unavailable.
                if len(results) == len(batch_df):
                    candidate = results.copy()
                    candidate[self._reference_column] = batch_df[column].to_numpy()
                    merged = candidate
                    break

        if merged is None and "metadata" in results.columns:
            # Some stages nest the expected answer inside metadata.
            def _extract_expected(meta: Any) -> Optional[str]:
                if isinstance(meta, Mapping):
                    for key in (self._reference_column, dataset_expected, "expected_answer", "gt", "label", "answer", "target"):
                        if key and key in meta:
                            return meta.get(key)
                return None

            candidate = results.copy()
            candidate[self._reference_column] = candidate["metadata"].apply(_extract_expected)
            if candidate[self._reference_column].notna().any():
                merged = candidate

        if merged is not None:
            merged[self._reference_column] = merged[self._reference_column].fillna("")
            return merged

        with open("/share/pierson/matt/mllmsci/gepa_debug.log", "a") as f:
            f.write(f"DEBUG: Merge failed. batch_df columns: {batch_df.columns.tolist()}\n")
            f.write(f"DEBUG: Merge failed. results columns: {results.columns.tolist()}\n")
            f.write(f"DEBUG: Candidate columns for merge: {candidate_columns}\n")
            f.write(f"DEBUG: batch_df shape: {batch_df.shape}, results shape: {results.shape}\n")
            if "sample_id" in results.columns and "sample_id" in batch_df.columns:
                 f.write("DEBUG: sample_id present in both\n")
            else:
                 f.write("DEBUG: sample_id missing from one or both\n")
        
        raise ValueError(
            f"Result DataFrame does not contain '{self._reference_column}' "
            "and no fallback column is available in the minibatch."
        )

    def _compute_metrics(self, frame: pd.DataFrame) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for metric_name in self._metric_order:
            spec = self._metric_specs.get(metric_name, {})
            metrics[metric_name] = self._compute_metric(frame, spec, metric_name)
        return metrics

    def _compute_metric(self, frame: pd.DataFrame, spec: Dict[str, Any], metric_name: str) -> float:
        if len(frame) == 0:
            return 0.0

        prediction_column = spec.get("prediction_column", self._prediction_column)
        reference_column = spec.get("reference_column", self._reference_column)
        # For binary metrics, allow specifying the positive class (default: "yes")
        positive_class = spec.get("positive_class", "yes").lower()
        negative_class = "no"  # Assumed binary classification

        # Use lenient extraction for predictions (handle "Yes, because..." responses)
        raw_predictions = frame[prediction_column].fillna("").astype(str)
        predictions = raw_predictions.apply(
            lambda x: self._extract_binary_answer(x, positive_class, negative_class)
        )
        expected = frame[reference_column].fillna("").astype(str).str.strip().str.lower()

        key = metric_name.lower()
        if key in {"accuracy", "exact_match"}:
            return float((predictions == expected).mean())
        if key == "f1":
            return self._compute_macro_f1(predictions.tolist(), expected.tolist())
        if key in {"precision", "ppv", "positive_predictive_value"}:
            # P(y=1|ŷ=1) = TP / (TP + FP)
            return self._compute_precision(predictions.tolist(), expected.tolist(), positive_class)
        if key in {"false_omission_rate", "for", "miss_rate_given_negative"}:
            # P(y=1|ŷ=0) = FN / (FN + TN)
            return self._compute_false_omission_rate(predictions.tolist(), expected.tolist(), positive_class)
        if key in {"npv", "negative_predictive_value"}:
            # P(y=0|ŷ=0) = TN / (TN + FN) = 1 - FOR
            return 1.0 - self._compute_false_omission_rate(predictions.tolist(), expected.tolist(), positive_class)
        if key in {"recall", "sensitivity", "tpr"}:
            # P(ŷ=1|y=1) = TP / (TP + FN)
            return self._compute_recall(predictions.tolist(), expected.tolist(), positive_class)
        # Fallback to accuracy if unknown metric
        return float((predictions == expected).mean())

    @staticmethod
    def _compute_macro_f1(preds: List[str], refs: List[str]) -> float:
        if not preds:
            return 0.0
        labels = sorted(set(refs) | set(preds))
        if not labels:
            return 0.0

        f1_scores: List[float] = []
        for label in labels:
            tp = sum(1 for p, r in zip(preds, refs) if p == label and r == label)
            fp = sum(1 for p, r in zip(preds, refs) if p == label and r != label)
            fn = sum(1 for p, r in zip(preds, refs) if p != label and r == label)

            if tp == 0 and fp == 0 and fn == 0:
                f1_scores.append(1.0)
                continue
            if tp == 0:
                f1_scores.append(0.0)
                continue

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1_scores.append(2 * precision * recall / (precision + recall))

        if not f1_scores:
            return 0.0
        return float(sum(f1_scores) / len(f1_scores))

    @staticmethod
    def _compute_binary_confusion(
        preds: List[str], refs: List[str], positive_class: str
    ) -> tuple:
        """Compute binary confusion matrix components.
        
        Args:
            preds: Model predictions
            refs: Ground truth labels  
            positive_class: The label considered as positive (e.g., "yes")
            
        Returns:
            Tuple of (TP, FP, TN, FN) counts
        """
        positive_class = positive_class.lower()
        tp = sum(1 for p, r in zip(preds, refs) if p == positive_class and r == positive_class)
        fp = sum(1 for p, r in zip(preds, refs) if p == positive_class and r != positive_class)
        tn = sum(1 for p, r in zip(preds, refs) if p != positive_class and r != positive_class)
        fn = sum(1 for p, r in zip(preds, refs) if p != positive_class and r == positive_class)
        return tp, fp, tn, fn

    @staticmethod
    def _compute_precision(preds: List[str], refs: List[str], positive_class: str = "yes") -> float:
        """Compute precision: P(y=1|ŷ=1) = TP / (TP + FP).
        
        When the model predicts positive, how often is the ground truth actually positive?
        High precision means few false alarms.
        """
        tp, fp, _, _ = GEPAVQAAdapter._compute_binary_confusion(preds, refs, positive_class)
        if tp + fp == 0:
            return 0.0  # No positive predictions made
        return float(tp / (tp + fp))

    @staticmethod
    def _compute_false_omission_rate(preds: List[str], refs: List[str], positive_class: str = "yes") -> float:
        """Compute false omission rate: P(y=1|ŷ=0) = FN / (FN + TN).
        
        When the model predicts negative, how often is the ground truth actually positive?
        Low FOR is desirable - we don't want to miss actual positive cases.
        For flood detection: low FOR means we rarely miss actual floods.
        """
        _, _, tn, fn = GEPAVQAAdapter._compute_binary_confusion(preds, refs, positive_class)
        if fn + tn == 0:
            return 0.0  # No negative predictions made
        return float(fn / (fn + tn))

    @staticmethod
    def _compute_recall(preds: List[str], refs: List[str], positive_class: str = "yes") -> float:
        """Compute recall/sensitivity: P(ŷ=1|y=1) = TP / (TP + FN).
        
        Of all actual positive cases, how many did the model correctly identify?
        High recall means few missed positive cases.
        """
        tp, _, _, fn = GEPAVQAAdapter._compute_binary_confusion(preds, refs, positive_class)
        if tp + fn == 0:
            return 0.0  # No actual positive cases
        return float(tp / (tp + fn))

    def _build_traces(
        self,
        frame: pd.DataFrame,
        candidate: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        traces: List[Dict[str, Any]] = []
        for _, row in frame.iterrows():
            metadata = row.get("metadata") or {}
            if isinstance(metadata, MutableMapping):
                metadata = dict(metadata)
            
            # Include image reference for multimodal reflection
            # Priority: image_path > image_url > image_base64
            image_ref = (
                row.get("image_path") or 
                row.get("image_url") or 
                row.get("image_base64")
            )
            
            traces.append(
                {
                    "prompt": row.get("prompt"),
                    "answer": row.get(self._prediction_column),
                    "expected_answer": row.get(self._reference_column),
                    "model_response": row.get("model_response"),
                    "sample_id": row.get("sample_id"),
                    "candidate": dict(candidate),
                    "metadata": metadata,
                    "image_ref": image_ref,  # For multimodal reflection
                }
            )
        return traces

