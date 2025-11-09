# Decision Tree Prompts

Decision tree prompts enable adaptive and context-sensitive VQA interactions where the next question depends on previous answers.

## Overview

Decision trees organize questions in a tree structure where:
- Each **node** represents a question/prompt
- **Branches** represent possible answers or conditions that determine the next question
- **Convergence nodes** support many-to-one patterns where multiple analysis paths converge

## Tree Structure

### Node Types

- **question**: Standard question node that branches based on model response
- **decision**: Decision point with conditional branching
- **leaf**: Terminal node (ends traversal)
- **convergence**: Many-to-one node that aggregates inputs from multiple predecessor nodes

### Condition Types

- **keyword_match**: Matches if any keywords appear in response
- **regex_match**: Matches if regex pattern matches response
- **confidence_threshold**: Matches if confidence exceeds threshold
- **always**: Always matches (unconditional branch)
- **default**: Matches if no other conditions match

### Branch Properties

- **target_node**: Next node ID if condition matches
- **weight**: Priority/weight for branch selection (higher = evaluated first)

## Example Tree

See `urban_planning_vqa.yaml` for a complete example that demonstrates:
- Conditional branching based on land use type
- Sequential follow-up questions
- Many-to-one convergence for synthesis

## Usage

Enable decision tree prompts in your configuration:

```yaml
prompt:
  decision_tree:
    enabled: true
    tree_path: "dagspaces/urbanvqa/conf/prompts/decision_trees/urban_planning_vqa.yaml"
    tree_format: "yaml"
    max_depth: 10
    enable_cycle_detection: true
```

Or via command line:

```bash
python -m dagspaces.urbanvqa.cli \
  pipeline=vqa \
  prompt.decision_tree.enabled=true \
  'prompt.decision_tree.tree_path=dagspaces/urbanvqa/conf/prompts/decision_trees/urban_planning_vqa.yaml'
```

## Many-to-One Convergence

Convergence nodes aggregate outputs from multiple predecessor nodes:

```yaml
convergence_node:
  node_id: "synthesis"
  node_type: "convergence"
  metadata:
    aggregation_strategy: "summarize"  # or "concatenate", "list", "json"
    aggregation_prompt: |
      Based on:
      Analysis 1: {{analysis_1}}
      Analysis 2: {{analysis_2}}
      Provide synthesis.
  prompt: "Synthesize: {{aggregated_inputs}}"
  output_key: "final_answer"
```

## Output Format

When decision tree prompts are enabled, output includes all intermediate outputs:

```python
{
    "sample_id": "...",
    "prompt": "...",
    "primary_land_use": "...",  # Node 1 output
    "density_level": "...",      # Node 2 output
    "final_answer": "...",       # Final output
    "metadata": {
        "tree_id": "urban_planning_vqa_v1",
        "nodes_visited": ["start", "residential_analysis", ...],
        "depth": 3
    }
}
```
