"""GEPA Prompt Evolution Visualization.

This module provides comprehensive visualizations for GEPA prompt optimization runs:
1. Fitness convergence chart with early stopping markers
2. Prompt diff timeline showing text evolution
3. Sankey diagram showing candidate flow and pruning

Usage:
    # From W&B run
    python -m dagspaces.urbanvqa.prompt_opt.visualize --wandb-run urbanekg/URBANVQA/run_id
    
    # From local artifacts
    python -m dagspaces.urbanvqa.prompt_opt.visualize --artifact-dir outputs/gepa/bayflood
    
    # Generate all visualizations
    python -m dagspaces.urbanvqa.prompt_opt.visualize --wandb-run ... --output-dir viz_output
"""

from __future__ import annotations

import argparse
import difflib
import html
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

LOG = logging.getLogger(__name__)


@dataclass
class PromptCandidate:
    """Represents a prompt candidate at a specific iteration."""
    iteration: int
    prompt_text: str
    component_name: str
    val_score: Optional[float] = None
    subsample_score: Optional[float] = None
    status: str = "evaluated"  # improved, worse, no_change, pruned
    parent_iteration: Optional[int] = None
    is_seed: bool = False
    is_best: bool = False


@dataclass 
class EvolutionHistory:
    """Complete history of prompt evolution."""
    candidates: List[PromptCandidate] = field(default_factory=list)
    seed_prompt: Optional[str] = None
    best_prompt: Optional[str] = None
    best_score: Optional[float] = None
    component_name: str = "user_prompt"
    total_iterations: int = 0
    early_stopped: bool = False
    early_stop_iteration: Optional[int] = None
    
    @classmethod
    def from_wandb(cls, run_path: str) -> "EvolutionHistory":
        """Load evolution history from a W&B run."""
        try:
            import wandb
        except ImportError:
            raise ImportError("wandb required: pip install wandb")
        
        api = wandb.Api()
        run = api.run(run_path)
        
        history = cls()
        
        # Get evolution history table
        tables = [a for a in run.logged_artifacts() if "evolution_history" in a.name]
        if tables:
            artifact = tables[-1]
            table = artifact.get("gepa/evolution_history")
            if table:
                df = pd.DataFrame(data=table.data, columns=table.columns)
                history._load_from_dataframe(df)
        
        # Fallback: scan run history
        if not history.candidates:
            history._load_from_run_history(run)
        
        # Get final metrics from run summary
        if run.summary:
            # Try multiple possible key names for best score
            history.best_score = (
                run.summary.get("gepa/final_best_val_score") or
                run.summary.get("gepa/best_score") or
                run.summary.get("gepa/best_val_score")
            )
        
        # If still no best_score, compute from candidates
        if history.best_score is None and history.candidates:
            scores = [c.val_score for c in history.candidates if c.val_score is not None]
            if scores:
                history.best_score = max(scores)
            
        return history
    
    @classmethod
    def from_artifacts(cls, artifact_dir: str | Path) -> "EvolutionHistory":
        """Load evolution history from local GEPA artifacts."""
        artifact_dir = Path(artifact_dir)
        history = cls()
        
        # Load best prompts
        best_prompts_path = artifact_dir / "best_prompts.yaml"
        if best_prompts_path.exists():
            import yaml
            with open(best_prompts_path) as f:
                best_prompts = yaml.safe_load(f) or {}
            history.best_prompt = best_prompts.get("user_prompt", "")
            history.component_name = list(best_prompts.keys())[0] if best_prompts else "user_prompt"
        
        # Load metrics
        metrics_path = artifact_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            history.best_score = metrics.get("best_score")
        
        # Load evolution CSV if exists (written by enhanced logging)
        evolution_csv = artifact_dir / "evolution_history.csv"
        if evolution_csv.exists():
            df = pd.read_csv(evolution_csv)
            history._load_from_dataframe(df)
        
        # Load traces for additional detail
        traces_path = artifact_dir / "traces.jsonl"
        if traces_path.exists():
            history._load_from_traces(traces_path)
            
        return history
    
    def _load_from_dataframe(self, df: pd.DataFrame) -> None:
        """Load candidates from evolution history DataFrame."""
        prompt_cols = [c for c in df.columns if c.startswith("prompt_")]
        if prompt_cols:
            self.component_name = prompt_cols[0].replace("prompt_", "")
        
        best_val_so_far = -float("inf")
        
        for _, row in df.iterrows():
            iteration = int(row.get("iteration", 0))
            prompt_text = row.get(f"prompt_{self.component_name}", "")
            if pd.isna(prompt_text):
                prompt_text = ""
            
            val_score = row.get("val_score")
            if pd.isna(val_score):
                val_score = row.get("best_val_score")
            
            status = row.get("status", "evaluated")
            is_seed = bool(row.get("is_seed", False))
            
            # Track if this became the new best
            is_best = False
            if val_score is not None and not pd.isna(val_score):
                if val_score > best_val_so_far:
                    best_val_so_far = val_score
                    is_best = True
            
            candidate = PromptCandidate(
                iteration=iteration,
                prompt_text=str(prompt_text),
                component_name=self.component_name,
                val_score=float(val_score) if val_score is not None and not pd.isna(val_score) else None,
                subsample_score=float(row.get("subsample_score_after")) if pd.notna(row.get("subsample_score_after")) else None,
                status=str(status) if pd.notna(status) else "evaluated",
                parent_iteration=iteration - 1 if iteration > 0 else None,
                is_seed=is_seed,
                is_best=is_best,
            )
            self.candidates.append(candidate)
        
        self.total_iterations = len(self.candidates)
        
        # Set seed prompt
        seed_candidates = [c for c in self.candidates if c.is_seed]
        if seed_candidates:
            self.seed_prompt = seed_candidates[0].prompt_text
        
        # Find best prompt
        best_candidates = sorted(
            [c for c in self.candidates if c.val_score is not None],
            key=lambda c: c.val_score or 0,
            reverse=True
        )
        if best_candidates:
            self.best_prompt = best_candidates[0].prompt_text
            self.best_score = best_candidates[0].val_score
    
    def _load_from_run_history(self, run) -> None:
        """Load from W&B run history (fallback)."""
        history_df = run.history()
        
        # Look for GEPA metrics in history
        gepa_cols = [c for c in history_df.columns if c.startswith("gepa/")]
        if not gepa_cols:
            return
        
        # Extract iteration data
        for idx, row in history_df.iterrows():
            iteration = row.get("gepa/iteration") or row.get("_step", idx)
            val_score = row.get("gepa/val_program_average") or row.get("gepa/best_valset_agg_score")
            
            # Look for prompt text
            prompt_text = ""
            for col in history_df.columns:
                if "new_instruction" in col.lower() or "prompt" in col.lower():
                    val = row.get(col)
                    if pd.notna(val) and val:
                        prompt_text = str(val)
                        break
            
            if val_score is not None and not pd.isna(val_score):
                candidate = PromptCandidate(
                    iteration=int(iteration),
                    prompt_text=prompt_text,
                    component_name=self.component_name,
                    val_score=float(val_score),
                    is_seed=(idx == 0),
                )
                self.candidates.append(candidate)
        
        self.total_iterations = len(self.candidates)
    
    def _load_from_traces(self, traces_path: Path) -> None:
        """Load additional detail from traces file."""
        with open(traces_path) as f:
            for line in f:
                try:
                    trace = json.loads(line)
                    # Could extract more detailed trajectory info here
                except json.JSONDecodeError:
                    continue


def generate_fitness_chart(
    history: EvolutionHistory,
    output_path: Optional[Path] = None,
    title: str = "GEPA Prompt Evolution - Fitness Over Iterations"
) -> str:
    """Generate interactive fitness convergence chart with Plotly.
    
    Shows:
    - Best score line (cumulative best)
    - Current iteration score
    - Early stopping point marker
    - Annotations for significant prompt changes
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("plotly required: pip install plotly")
    
    if not history.candidates:
        return "<p>No evolution data available</p>"
    
    # Prepare data
    iterations = []
    val_scores = []
    best_scores = []
    statuses = []
    prompts = []
    
    best_so_far = -float("inf")
    for c in sorted(history.candidates, key=lambda x: x.iteration):
        iterations.append(c.iteration)
        val_scores.append(c.val_score)
        
        if c.val_score is not None and c.val_score > best_so_far:
            best_so_far = c.val_score
        best_scores.append(best_so_far if best_so_far > -float("inf") else None)
        
        statuses.append(c.status)
        prompts.append(c.prompt_text[:100] + "..." if len(c.prompt_text) > 100 else c.prompt_text)
    
    # Create figure
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=("Validation Score", "Score Delta from Previous"),
        vertical_spacing=0.12
    )
    
    # Main score line
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=val_scores,
            mode="lines+markers",
            name="Current Score",
            line=dict(color="#636EFA", width=2),
            marker=dict(size=8),
            hovertemplate="<b>Iteration %{x}</b><br>Score: %{y:.4f}<br>Prompt: %{customdata}<extra></extra>",
            customdata=prompts,
        ),
        row=1, col=1
    )
    
    # Best score line
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=best_scores,
            mode="lines",
            name="Best So Far",
            line=dict(color="#00CC96", width=3, dash="dash"),
            hovertemplate="<b>Iteration %{x}</b><br>Best Score: %{y:.4f}<extra></extra>",
        ),
        row=1, col=1
    )
    
    # Mark improvements
    improvement_iters = []
    improvement_scores = []
    improvement_texts = []
    
    for i, c in enumerate(sorted(history.candidates, key=lambda x: x.iteration)):
        if c.is_best and c.val_score is not None:
            improvement_iters.append(c.iteration)
            improvement_scores.append(c.val_score)
            improvement_texts.append(f"New Best: {c.val_score:.4f}")
    
    if improvement_iters:
        fig.add_trace(
            go.Scatter(
                x=improvement_iters,
                y=improvement_scores,
                mode="markers",
                name="New Best Found",
                marker=dict(color="#EF553B", size=14, symbol="star"),
                hovertemplate="<b>New Best!</b><br>Iteration %{x}<br>Score: %{y:.4f}<extra></extra>",
            ),
            row=1, col=1
        )
    
    # Delta subplot
    deltas = [0]
    for i in range(1, len(val_scores)):
        if val_scores[i] is not None and val_scores[i-1] is not None:
            deltas.append(val_scores[i] - val_scores[i-1])
        else:
            deltas.append(0)
    
    colors = ["#00CC96" if d > 0 else "#EF553B" if d < 0 else "#636EFA" for d in deltas]
    
    fig.add_trace(
        go.Bar(
            x=iterations,
            y=deltas,
            name="Score Delta",
            marker_color=colors,
            hovertemplate="<b>Iteration %{x}</b><br>Delta: %{y:.4f}<extra></extra>",
        ),
        row=2, col=1
    )
    
    # Add zero line to delta subplot
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    # Layout
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        height=700,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        template="plotly_white",
    )
    
    fig.update_xaxes(title_text="Iteration", row=2, col=1)
    fig.update_yaxes(title_text="F1 Score", row=1, col=1)
    fig.update_yaxes(title_text="Δ Score", row=2, col=1)
    
    # Add annotation for best score
    if history.best_score:
        fig.add_annotation(
            x=iterations[-1] if iterations else 0,
            y=history.best_score,
            text=f"Best: {history.best_score:.4f}",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#00CC96",
            font=dict(color="#00CC96", size=12),
            row=1, col=1
        )
    
    html_str = fig.to_html(full_html=False, include_plotlyjs="cdn")
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head><title>{title}</title></head>
<body>{html_str}</body>
</html>""")
        LOG.info("Fitness chart saved to %s", output_path)
    
    return html_str


def generate_prompt_diff_timeline(
    history: EvolutionHistory,
    output_path: Optional[Path] = None,
    title: str = "Prompt Evolution Timeline"
) -> str:
    """Generate an HTML diff timeline showing how the prompt text evolved.
    
    Shows side-by-side or inline diffs between consecutive best prompts.
    """
    if not history.candidates:
        return "<p>No evolution data available</p>"
    
    # Get sequence of best prompts (only when score improved)
    best_sequence: List[Tuple[int, str, float]] = []
    best_score_so_far = -float("inf")
    
    for c in sorted(history.candidates, key=lambda x: x.iteration):
        if c.val_score is not None and c.val_score > best_score_so_far:
            best_score_so_far = c.val_score
            best_sequence.append((c.iteration, c.prompt_text, c.val_score))
    
    # If no improvements tracked, use all candidates
    if len(best_sequence) <= 1:
        best_sequence = [
            (c.iteration, c.prompt_text, c.val_score or 0)
            for c in sorted(history.candidates, key=lambda x: x.iteration)
            if c.prompt_text
        ]
    
    css = """
    <style>
        .diff-timeline {
            font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #1e1e2e;
            color: #cdd6f4;
        }
        .diff-timeline h1 {
            color: #89b4fa;
            text-align: center;
            margin-bottom: 30px;
        }
        .iteration-card {
            background: #313244;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            border-left: 4px solid #89b4fa;
        }
        .iteration-card.improved {
            border-left-color: #a6e3a1;
        }
        .iteration-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        .iteration-number {
            font-size: 1.2em;
            font-weight: bold;
            color: #89b4fa;
        }
        .score-badge {
            background: #45475a;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.9em;
        }
        .score-badge.improved {
            background: #a6e3a1;
            color: #1e1e2e;
        }
        .prompt-text {
            background: #45475a;
            padding: 15px;
            border-radius: 8px;
            white-space: pre-wrap;
            word-wrap: break-word;
            line-height: 1.6;
        }
        .diff-section {
            margin-top: 15px;
        }
        .diff-label {
            font-size: 0.85em;
            color: #a6adc8;
            margin-bottom: 8px;
        }
        .diff-content {
            background: #11111b;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
        }
        .diff-add {
            background: rgba(166, 227, 161, 0.2);
            color: #a6e3a1;
        }
        .diff-del {
            background: rgba(243, 139, 168, 0.2);
            color: #f38ba8;
            text-decoration: line-through;
        }
        .diff-line {
            display: block;
            padding: 2px 5px;
        }
        .arrow-connector {
            text-align: center;
            font-size: 2em;
            color: #6c7086;
            margin: 10px 0;
        }
        .summary-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: #313244;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #89b4fa;
        }
        .stat-label {
            color: #a6adc8;
            font-size: 0.9em;
        }
    </style>
    """
    
    def make_inline_diff(old: str, new: str) -> str:
        """Create inline diff HTML."""
        if not old or not new:
            return html.escape(new or old or "")
        
        # Use word-level diff
        old_words = old.split()
        new_words = new.split()
        
        matcher = difflib.SequenceMatcher(None, old_words, new_words)
        result = []
        
        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op == "equal":
                result.append(" ".join(old_words[i1:i2]))
            elif op == "delete":
                deleted = " ".join(old_words[i1:i2])
                result.append(f'<span class="diff-del">{html.escape(deleted)}</span>')
            elif op == "insert":
                inserted = " ".join(new_words[j1:j2])
                result.append(f'<span class="diff-add">{html.escape(inserted)}</span>')
            elif op == "replace":
                deleted = " ".join(old_words[i1:i2])
                inserted = " ".join(new_words[j1:j2])
                result.append(f'<span class="diff-del">{html.escape(deleted)}</span>')
                result.append(f'<span class="diff-add">{html.escape(inserted)}</span>')
        
        return " ".join(result)
    
    # Build HTML
    best_score_str = f"{history.best_score:.4f}" if history.best_score else "N/A"
    
    html_parts = [f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    {css}
</head>
<body>
<div class="diff-timeline">
    <h1>{title}</h1>
    
    <div class="summary-stats">
        <div class="stat-card">
            <div class="stat-value">{len(best_sequence)}</div>
            <div class="stat-label">Prompt Versions</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{history.total_iterations}</div>
            <div class="stat-label">Total Iterations</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{best_score_str}</div>
            <div class="stat-label">Best Score</div>
        </div>
    </div>
"""]
    
    prev_text = ""
    for i, (iteration, prompt_text, score) in enumerate(best_sequence):
        is_first = i == 0
        improved = not is_first and score > best_sequence[i-1][2] if score else False
        
        card_class = "iteration-card improved" if improved else "iteration-card"
        badge_class = "score-badge improved" if improved else "score-badge"
        
        # Format score string
        score_str = f"{score:.4f}" if score else "N/A"
        
        # Score improvement
        delta = ""
        if not is_first and score and best_sequence[i-1][2]:
            d = score - best_sequence[i-1][2]
            delta = f" (+{d:.4f})" if d > 0 else f" ({d:.4f})"
        
        # Format iteration label
        iter_label = "Seed Prompt" if is_first else f"Iteration {iteration}"
        
        html_parts.append(f"""
    <div class="{card_class}">
        <div class="iteration-header">
            <span class="iteration-number">{iter_label}</span>
            <span class="{badge_class}">Score: {score_str}{delta}</span>
        </div>
        <div class="prompt-text">{html.escape(prompt_text)}</div>
""")
        
        # Add diff from previous
        if not is_first and prev_text:
            diff_html = make_inline_diff(prev_text, prompt_text)
            html_parts.append(f"""
        <div class="diff-section">
            <div class="diff-label">Changes from previous:</div>
            <div class="diff-content">{diff_html}</div>
        </div>
""")
        
        html_parts.append("    </div>")
        
        # Arrow between cards
        if i < len(best_sequence) - 1:
            html_parts.append('    <div class="arrow-connector">↓</div>')
        
        prev_text = prompt_text
    
    html_parts.append("""
</div>
</body>
</html>
""")
    
    html_str = "\n".join(html_parts)
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(html_str)
        LOG.info("Prompt diff timeline saved to %s", output_path)
    
    return html_str


def generate_sankey_diagram(
    history: EvolutionHistory,
    output_path: Optional[Path] = None,
    title: str = "Prompt Candidate Flow"
) -> str:
    """Generate Sankey diagram showing candidate flow and pruning.
    
    Shows:
    - How candidates flow through iterations
    - Which candidates were accepted vs pruned
    - Width proportional to score
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly required: pip install plotly")
    
    if not history.candidates:
        return "<p>No evolution data available</p>"
    
    # Group candidates by iteration
    by_iteration: Dict[int, List[PromptCandidate]] = {}
    for c in history.candidates:
        if c.iteration not in by_iteration:
            by_iteration[c.iteration] = []
        by_iteration[c.iteration].append(c)
    
    iterations = sorted(by_iteration.keys())
    
    if len(iterations) < 2:
        return "<p>Not enough iterations for Sankey diagram</p>"
    
    # Build Sankey data
    labels = []
    colors = []
    node_x = []
    node_y = []
    
    sources = []
    targets = []
    values = []
    link_colors = []
    
    node_idx = 0
    iteration_nodes: Dict[int, Dict[str, int]] = {}  # iteration -> {status: node_idx}
    
    # Create nodes for each iteration
    for i, iter_num in enumerate(iterations):
        candidates = by_iteration[iter_num]
        iteration_nodes[iter_num] = {}
        
        # Group by status
        improved = [c for c in candidates if c.status == "improved" or c.is_best]
        maintained = [c for c in candidates if c.status in ("evaluated", "no_change") and not c.is_best]
        rejected = [c for c in candidates if c.status == "worse"]
        
        x_pos = i / max(len(iterations) - 1, 1)
        
        # Add nodes
        if improved:
            best_score = max(c.val_score or 0 for c in improved)
            labels.append(f"Iter {iter_num}\nImproved\n({best_score:.3f})")
            colors.append("rgba(166, 227, 161, 0.8)")  # Green
            node_x.append(x_pos)
            node_y.append(0.2)
            iteration_nodes[iter_num]["improved"] = node_idx
            node_idx += 1
        
        if maintained:
            labels.append(f"Iter {iter_num}\nMaintained")
            colors.append("rgba(137, 180, 250, 0.8)")  # Blue
            node_x.append(x_pos)
            node_y.append(0.5)
            iteration_nodes[iter_num]["maintained"] = node_idx
            node_idx += 1
        
        if rejected:
            labels.append(f"Iter {iter_num}\nRejected")
            colors.append("rgba(243, 139, 168, 0.6)")  # Red, slightly transparent
            node_x.append(x_pos)
            node_y.append(0.8)
            iteration_nodes[iter_num]["rejected"] = node_idx
            node_idx += 1
    
    # Create links between iterations
    for i in range(len(iterations) - 1):
        curr_iter = iterations[i]
        next_iter = iterations[i + 1]
        
        curr_nodes = iteration_nodes.get(curr_iter, {})
        next_nodes = iteration_nodes.get(next_iter, {})
        
        # Best/improved flows forward
        if "improved" in curr_nodes:
            if "improved" in next_nodes:
                sources.append(curr_nodes["improved"])
                targets.append(next_nodes["improved"])
                values.append(1)
                link_colors.append("rgba(166, 227, 161, 0.4)")
            elif "maintained" in next_nodes:
                sources.append(curr_nodes["improved"])
                targets.append(next_nodes["maintained"])
                values.append(1)
                link_colors.append("rgba(166, 227, 161, 0.4)")
        
        # Maintained flows forward
        if "maintained" in curr_nodes:
            if "improved" in next_nodes:
                sources.append(curr_nodes["maintained"])
                targets.append(next_nodes["improved"])
                values.append(0.5)
                link_colors.append("rgba(137, 180, 250, 0.3)")
            elif "maintained" in next_nodes:
                sources.append(curr_nodes["maintained"])
                targets.append(next_nodes["maintained"])
                values.append(0.5)
                link_colors.append("rgba(137, 180, 250, 0.3)")
            if "rejected" in next_nodes:
                sources.append(curr_nodes["maintained"])
                targets.append(next_nodes["rejected"])
                values.append(0.3)
                link_colors.append("rgba(243, 139, 168, 0.2)")
    
    # Handle case with no links
    if not sources:
        # Create simple linear flow
        for i in range(len(labels) - 1):
            sources.append(i)
            targets.append(i + 1)
            values.append(1)
            link_colors.append("rgba(137, 180, 250, 0.4)")
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=30,
            line=dict(color="white", width=1),
            label=labels,
            color=colors,
            x=node_x if node_x else None,
            y=node_y if node_y else None,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
        ),
    )])
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        font=dict(size=12, family="SF Mono, Consolas, monospace"),
        height=500,
        template="plotly_white",
    )
    
    html_str = fig.to_html(full_html=False, include_plotlyjs="cdn")
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head><title>{title}</title></head>
<body>{html_str}</body>
</html>""")
        LOG.info("Sankey diagram saved to %s", output_path)
    
    return html_str


def generate_combined_dashboard(
    history: EvolutionHistory,
    output_path: Path,
    title: str = "GEPA Prompt Evolution Dashboard"
) -> None:
    """Generate a combined HTML dashboard with all visualizations."""
    
    # Pre-compute formatted strings to avoid f-string issues
    best_score_str = f"{history.best_score:.4f}" if history.best_score else "N/A"
    
    fitness_html = generate_fitness_chart(history)
    diff_timeline_html = generate_prompt_diff_timeline(history)
    sankey_html = generate_sankey_diagram(history)
    
    # Extract just the body content from diff timeline
    diff_match = re.search(r'<div class="diff-timeline">.*?</div>\s*</body>', diff_timeline_html, re.DOTALL)
    diff_body = diff_match.group(0).replace('</body>', '') if diff_match else diff_timeline_html
    
    # Extract CSS from diff timeline
    css_match = re.search(r'<style>.*?</style>', diff_timeline_html, re.DOTALL)
    diff_css = css_match.group(0) if css_match else ""
    
    dashboard_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    {diff_css}
    <style>
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #0f0f1a;
            color: #e0e0e0;
            margin: 0;
            padding: 0;
        }}
        .dashboard-header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 30px;
            text-align: center;
            border-bottom: 1px solid #2a2a4a;
        }}
        .dashboard-header h1 {{
            margin: 0;
            font-size: 2.5em;
            background: linear-gradient(90deg, #89b4fa, #a6e3a1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .dashboard-header .subtitle {{
            color: #6c7086;
            margin-top: 10px;
        }}
        .dashboard-container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 30px;
        }}
        .section {{
            background: #1e1e2e;
            border-radius: 16px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }}
        .section-title {{
            font-size: 1.5em;
            color: #89b4fa;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #313244;
        }}
        .tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }}
        .tab {{
            padding: 10px 20px;
            background: #313244;
            border: none;
            border-radius: 8px;
            color: #cdd6f4;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .tab:hover {{
            background: #45475a;
        }}
        .tab.active {{
            background: #89b4fa;
            color: #1e1e2e;
        }}
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
        }}
        .best-prompt-box {{
            background: #313244;
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
        }}
        .best-prompt-label {{
            color: #a6e3a1;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .best-prompt-text {{
            font-family: 'SF Mono', monospace;
            background: #1e1e2e;
            padding: 15px;
            border-radius: 8px;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
    </style>
</head>
<body>
    <div class="dashboard-header">
        <h1>{title}</h1>
        <div class="subtitle">
            {history.total_iterations} iterations | 
            Best Score: {best_score_str} |
            Component: {history.component_name}
        </div>
    </div>
    
    <div class="dashboard-container">
        <!-- Fitness Chart Section -->
        <div class="section">
            <div class="section-title">📈 Fitness Convergence</div>
            {fitness_html}
        </div>
        
        <!-- Sankey Flow Section -->
        <div class="section">
            <div class="section-title">🔀 Candidate Flow</div>
            {sankey_html}
        </div>
        
        <!-- Prompt Evolution Section -->
        <div class="section">
            <div class="section-title">📝 Prompt Evolution Timeline</div>
            {diff_body}
        </div>
        
        <!-- Best Prompt Section -->
        <div class="section">
            <div class="section-title">🏆 Final Best Prompt</div>
            <div class="best-prompt-box">
                <div class="best-prompt-label">Score: {best_score_str}</div>
                <div class="best-prompt-text">{html.escape(history.best_prompt or 'No best prompt found')}</div>
            </div>
        </div>
    </div>
    
    <script>
        // Tab switching
        document.querySelectorAll('.tab').forEach(tab => {{
            tab.addEventListener('click', () => {{
                const tabGroup = tab.closest('.tabs').dataset.group;
                document.querySelectorAll(`[data-group="${{tabGroup}}"] .tab`).forEach(t => t.classList.remove('active'));
                document.querySelectorAll(`.tab-content[data-group="${{tabGroup}}"]`).forEach(c => c.classList.remove('active'));
                tab.classList.add('active');
                document.querySelector(`.tab-content[data-tab="${{tab.dataset.tab}}"]`).classList.add('active');
            }});
        }});
    </script>
</body>
</html>
"""
    
    with open(output_path, "w") as f:
        f.write(dashboard_html)
    
    LOG.info("Combined dashboard saved to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Visualize GEPA prompt evolution")
    parser.add_argument("--wandb-run", type=str, help="W&B run path (entity/project/run_id)")
    parser.add_argument("--artifact-dir", type=str, help="Local artifact directory")
    parser.add_argument("--output-dir", type=str, default="viz_output", help="Output directory")
    parser.add_argument("--format", choices=["dashboard", "separate", "both"], default="dashboard",
                        help="Output format: combined dashboard or separate files")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if not args.wandb_run and not args.artifact_dir:
        parser.error("Either --wandb-run or --artifact-dir is required")
    
    # Load history
    if args.wandb_run:
        LOG.info("Loading from W&B run: %s", args.wandb_run)
        history = EvolutionHistory.from_wandb(args.wandb_run)
    else:
        LOG.info("Loading from artifacts: %s", args.artifact_dir)
        history = EvolutionHistory.from_artifacts(args.artifact_dir)
    
    LOG.info("Loaded %d candidates", len(history.candidates))
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    if args.format in ("dashboard", "both"):
        generate_combined_dashboard(history, output_dir / "dashboard.html")
    
    if args.format in ("separate", "both"):
        generate_fitness_chart(history, output_dir / "fitness_chart.html")
        generate_prompt_diff_timeline(history, output_dir / "prompt_timeline.html")
        generate_sankey_diagram(history, output_dir / "sankey_flow.html")
    
    LOG.info("Visualization complete! Output in: %s", output_dir)


if __name__ == "__main__":
    main()

