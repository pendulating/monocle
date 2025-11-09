
## Phase 6: Future Enhancements (Planned)

### 6.1 Dynamic Prompts with Jinja2 Templates

**Goal**: Support Jinja2-style template rendering for dynamic prompt generation with variable substitution.

**Configuration**:
```yaml
prompt:
  template: "Analyze this image focusing on {{focus_area}}. Question: {{user_question}}"
  template_vars:
    focus_area: "urban planning"
    analysis_depth: "detailed"
```

**Data Format**:
```python
{
    "prompt": "What type of building is this?",  # Base question
    "focus_area": "urban planning",  # Template variable
    "analysis_depth": "detailed",  # Template variable
    "image_path": "/path/to/image.jpg"
}
```

**Template Examples**:
```yaml
# Simple variable substitution
prompt:
  template: "Analyze this image focusing on {{focus_area}}. Question: {{prompt}}"

# Conditional rendering
prompt:
  template: |
    {% if focus_area %}
    Analyze this image focusing on {{focus_area}}.
    {% else %}
    Analyze this image.
    {% endif %}
    Question: {{prompt}}

# Loop over items
prompt:
  template: |
    Analyze this image considering:
    {% for criterion in criteria %}
    - {{criterion}}
    {% endfor %}
    Question: {{prompt}}

# Complex template with filters
prompt:
  template: |
    {{"Analyze this " + image_type|upper + " image."}}
    Focus areas: {{focus_areas|join(", ")}}
    Question: {{prompt}}
```

**Implementation** (Jinja2 Templates):
```python
from jinja2 import Template, Environment, StrictUndefined

def render_prompt_template(template_str: str, context: Dict[str, Any]) -> str:
    """Render Jinja2 template with variable substitution."""
    env = Environment(undefined=StrictUndefined)  # Raise error on undefined variables
    template = env.from_string(template_str)
    return template.render(**context)

# Usage in preprocessing
def preprocess_with_template(row: Dict[str, Any], cfg: DictConfig) -> Dict[str, Any]:
    """Preprocess with Jinja2 template support."""
    # Get template from config or row
    template_str = cfg.prompt.get("template") or row.get("prompt_template") or row.get("prompt")
    
    # Build context from row data and config
    context = {
        "prompt": row.get("prompt", ""),
        "user_question": row.get("prompt", ""),
        **row  # Include all row data as template variables
    }
    
    # Add config variables
    if hasattr(cfg.prompt, "template_vars"):
        context.update(cfg.prompt.template_vars)
    
    # Render template
    rendered_prompt = render_prompt_template(template_str, context)
    
    # Build messages
    image_content = _prepare_image_content(row.get("image"), row.get("sample_id"))
    messages = [
        {"role": "system", "content": cfg.prompt.system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": rendered_prompt},
                image_content
            ]
        }
    ]
    
    return {
        "messages": messages,
        "sampling_params": cfg.sampling_params_vqa
    }
```

**Metadata Support** (Recent Enhancement - see `implementation_09_recent-enhancements.md`):
- All metadata columns from parquet files are automatically available in Jinja2 templates
- Metadata columns are included in the template context via `**row` spread in `unified_preprocess()`
- Example: If parquet contains `location`, `timestamp`, `camera_id` columns, they can be used as `{{location}}`, `{{timestamp}}`, `{{camera_id}}` in templates
- Metadata is preserved through preprocessing/postprocessing stages
- Metadata columns are filtered to exclude only large/complex objects (images, arrays, PIL objects)

**Template Features**:
- Jinja2 syntax: `{{variable}}`, `{% if condition %}...{% endif %}`, `{% for item in list %}...{% endfor %}`
- Variable access: `{{prompt}}`, `{{sample_id}}`, `{{user_question}}`
- Conditional logic: `{% if focus_area %}{{focus_area}}{% endif %}`
- Loops: `{% for item in items %}{{item}}{% endfor %}`
- Filters: `{{variable|upper}}`, `{{variable|length}}`
- Strict mode: Raises error on undefined variables (prevents silent failures)

### 6.2 Hierarchical Prompts

**Goal**: Multi-step reasoning with intermediate outputs.

**Example Flow**:
1. **Observation Step**: "Describe what you see in this image."
2. **Reasoning Step**: "Based on your observation, analyze the urban planning implications."
3. **Answer Step**: "Finally, answer: What type of development is this?"

**Many-to-One Support**: Multiple parallel steps can feed into a single downstream step, enabling fan-in patterns where multiple analysis paths converge.

**Configuration**:
```yaml
prompt:
  hierarchical:
    enabled: true
    steps:
      - name: "observation"
        prompt: "Describe what you see in this image."
        output_key: "observation"
      - name: "reasoning"
        prompt: "Based on your observation: {{observation}}, analyze {{analysis_focus}}."
        output_key: "reasoning"
      - name: "answer"
        prompt: "Finally, answer: {{final_question}}"
        output_key: "answer"
```

**Many-to-One Configuration Example**:
```yaml
prompt:
  hierarchical:
    enabled: true
    steps:
      # Parallel analysis steps (many-to-one pattern)
      - name: "structure_analysis"
        prompt: "Identify and describe all structures visible in this image."
        output_key: "structures"
        parallel: true  # Mark as parallel step
      - name: "context_analysis"
        prompt: "Describe the surrounding context and environment."
        output_key: "context"
        parallel: true  # Mark as parallel step
      - name: "infrastructure_analysis"
        prompt: "Identify infrastructure elements (roads, utilities, etc.)."
        output_key: "infrastructure"
        parallel: true  # Mark as parallel step
      
      # Convergent step (receives all parallel outputs)
      - name: "synthesis"
        prompt: |
          Based on the following analyses:
          
          Structures: {{structures}}
          Context: {{context}}
          Infrastructure: {{infrastructure}}
          
          Provide a comprehensive urban planning assessment.
        output_key: "final_answer"
        depends_on: ["structures", "context", "infrastructure"]  # Many-to-one: multiple inputs
```

**Output Format**:
```python
{
    "sample_id": str,
    "observation": str,  # Step 1 output
    "reasoning": str,  # Step 2 output
    "answer": str,  # Final answer
    "metadata": dict,
}
```

**Many-to-One Output Format**:
```python
{
    "sample_id": str,
    "structures": str,  # Parallel step 1 output
    "context": str,  # Parallel step 2 output
    "infrastructure": str,  # Parallel step 3 output
    "final_answer": str,  # Convergent step output (uses all above)
    "metadata": dict,
}
```

**Implementation** (Ray Data + vLLM compatible):
```python
def preprocess_hierarchical_many_to_one(row: Dict[str, Any], cfg: DictConfig) -> Dict[str, Any]:
    """Preprocess hierarchical prompts with many-to-one support."""
    if not cfg.prompt.hierarchical.enabled:
        return preprocess_simple(row, cfg)
    
    steps = cfg.prompt.hierarchical.steps
    results = {}
    image_content = _prepare_image_content(row.get("image"), row.get("sample_id"))
    
    # Group steps by execution order
    parallel_groups = []
    current_group = []
    
    for step in steps:
        if step.get("parallel", False):
            current_group.append(step)
        else:
            if current_group:
                parallel_groups.append(current_group)
                current_group = []
            parallel_groups.append([step])
    
    if current_group:
        parallel_groups.append(current_group)
    
    # Execute steps in groups
    for group in parallel_groups:
        if len(group) == 1:
            # Single step (sequential or convergent)
            step = group[0]
            step_prompt = step.prompt
            
            # Replace placeholders with previous results
            for key, value in results.items():
                step_prompt = step_prompt.replace(f"{{{{{key}}}}}", str(value))
            
            # Add user question if placeholder exists
            if "{{final_question}}" in step_prompt or "{{user_question}}" in step_prompt:
                step_prompt = step_prompt.replace("{{final_question}}", row.get("prompt", ""))
                step_prompt = step_prompt.replace("{{user_question}}", row.get("prompt", ""))
            
            messages = [
                {"role": "system", "content": cfg.prompt.system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": step_prompt},
                        image_content
                    ]
                }
            ]
            
            # Call vLLM (in Ray Data, this happens via processor)
            # For now, return messages - actual call happens in processor
            return {
                "messages": messages,
                "sampling_params": cfg.sampling_params_vqa,
                "_hierarchical_step": step.name,
                "_hierarchical_output_key": step.output_key,
                "_hierarchical_results": results,  # Previous results
                "_hierarchical_depends_on": step.get("depends_on", [])
            }
        else:
            # Parallel group - execute all steps simultaneously
            # In Ray Data, we'd use map_batches with multiple calls
            # For preprocessing, we prepare messages for all parallel steps
            parallel_messages = []
            for step in group:
                step_prompt = step.prompt
                for key, value in results.items():
                    step_prompt = step_prompt.replace(f"{{{{{key}}}}}", str(value))
                
                messages = [
                    {"role": "system", "content": cfg.prompt.system},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": step_prompt},
                            image_content
                        ]
                    }
                ]
                parallel_messages.append({
                    "step_name": step.name,
                    "output_key": step.output_key,
                    "messages": messages
                })
            
            # Return first parallel step's messages
            # Postprocessing will handle collecting all parallel results
            return {
                "messages": parallel_messages[0]["messages"],
                "sampling_params": cfg.sampling_params_vqa,
                "_hierarchical_parallel_group": [
                    {"name": pm["step_name"], "output_key": pm["output_key"]}
                    for pm in parallel_messages
                ],
                "_hierarchical_results": results
            }
    
    # Fallback
    return preprocess_simple(row, cfg)
```

### 6.3 Decision Tree Prompts

**Goal**: Structured branching prompts where the next question depends on previous answers, enabling adaptive and context-sensitive VQA interactions.

**Description**: Decision tree prompts organize questions in a tree structure where each node represents a question/prompt, and branches represent possible answers or conditions that determine the next question. This enables:
- **Adaptive questioning**: Different questions based on image content or previous answers
- **Context-aware reasoning**: Branching logic guides the model through relevant follow-up questions
- **Efficient task adaptation**: No fine-tuning required; structure guides the model behavior

**Research Reference**: Tree prompting has been shown to efficiently adapt tasks without fine-tuning (ACL 2023: "Tree Prompting: Efficient Task Adaptation without Fine-Tuning").

#### 6.3.1 Decision Tree Storage Format

**Best Practice**: Use JSON schema for tree structure (most flexible and widely supported).

**JSON Schema**:
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "tree_id": {"type": "string", "description": "Unique identifier for the tree"},
    "version": {"type": "string", "description": "Tree version"},
    "root_node": {"type": "string", "description": "ID of the root node"},
    "nodes": {
      "type": "object",
      "patternProperties": {
        "^[a-zA-Z0-9_-]+$": {
          "type": "object",
          "properties": {
            "node_id": {"type": "string"},
            "prompt": {"type": "string", "description": "Question/prompt for this node"},
            "node_type": {
              "type": "string",
              "enum": ["question", "decision", "leaf", "action"],
              "description": "Type of node"
            },
            "branches": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "condition": {
                    "type": "object",
                    "description": "Condition to evaluate (e.g., keyword match, confidence threshold)"
                  },
                  "target_node": {"type": "string", "description": "Next node ID if condition matches"},
                  "weight": {"type": "number", "description": "Priority/weight for branch selection"}
                },
                "required": ["target_node"]
              }
            },
            "output_key": {"type": "string", "description": "Key to store output in results"},
            "metadata": {"type": "object", "description": "Additional node metadata"}
          },
          "required": ["node_id", "prompt", "node_type"]
        }
      }
    }
  },
  "required": ["tree_id", "root_node", "nodes"]
}
```

**YAML Example** (human-readable alternative):
```yaml
tree_id: "urban_planning_vqa_v1"
version: "1.0.0"
root_node: "start"
nodes:
  start:
    node_id: "start"
    prompt: "What is the primary land use visible in this image?"
    node_type: "question"
    output_key: "primary_land_use"
    branches:
      - condition:
          type: "keyword_match"
          keywords: ["residential", "housing", "apartments", "houses"]
        target_node: "residential_analysis"
        weight: 1.0
      - condition:
          type: "keyword_match"
          keywords: ["commercial", "retail", "shopping", "business"]
        target_node: "commercial_analysis"
        weight: 1.0
      - condition:
          type: "keyword_match"
          keywords: ["industrial", "factory", "warehouse", "manufacturing"]
        target_node: "industrial_analysis"
        weight: 1.0
      - condition:
          type: "default"
        target_node: "general_analysis"
        weight: 0.5
  
  residential_analysis:
    node_id: "residential_analysis"
    prompt: "Based on the residential context, what is the density level? (low, medium, high)"
    node_type: "question"
    output_key: "density_level"
    branches:
      - condition:
          type: "keyword_match"
          keywords: ["low", "low-density", "sparse"]
        target_node: "low_density_details"
      - condition:
          type: "keyword_match"
          keywords: ["medium", "medium-density"]
        target_node: "medium_density_details"
      - condition:
          type: "keyword_match"
          keywords: ["high", "high-density", "dense"]
        target_node: "high_density_details"
      - condition:
          type: "default"
        target_node: "convergence_assessment"
  
  commercial_analysis:
    node_id: "commercial_analysis"
    prompt: "What type of commercial activity is visible? (retail, office, mixed-use)"
    node_type: "question"
    output_key: "commercial_type"
    branches:
      - condition:
          type: "always"
        target_node: "convergence_assessment"
  
  industrial_analysis:
    node_id: "industrial_analysis"
    prompt: "What industrial characteristics are present?"
    node_type: "question"
    output_key: "industrial_chars"
    branches:
      - condition:
          type: "always"
        target_node: "convergence_assessment"
  
  low_density_details:
    node_id: "low_density_details"
    prompt: "Describe the specific characteristics of this low-density residential area."
    node_type: "question"
    output_key: "residential_details"
    branches:
      - condition:
          type: "always"
        target_node: "convergence_assessment"
  
  medium_density_details:
    node_id: "medium_density_details"
    prompt: "Describe the specific characteristics of this medium-density residential area."
    node_type: "question"
    output_key: "residential_details"
    branches:
      - condition:
          type: "always"
        target_node: "convergence_assessment"
  
  high_density_details:
    node_id: "high_density_details"
    prompt: "Describe the specific characteristics of this high-density residential area."
    node_type: "question"
    output_key: "residential_details"
    branches:
      - condition:
          type: "always"
        target_node: "convergence_assessment"
  
  # Many-to-one convergence node: aggregates outputs from multiple paths
  convergence_assessment:
    node_id: "convergence_assessment"
    node_type: "convergence"
    aggregation_strategy: "summarize"  # How to combine multiple inputs
    aggregation_prompt: |
      Based on the following analyses:
      
      Primary Land Use: {{primary_land_use}}
      Density Level: {{density_level}}
      Residential Details: {{residential_details}}
      Commercial Type: {{commercial_type}}
      Industrial Characteristics: {{industrial_chars}}
      
      Provide a comprehensive urban planning assessment.
    prompt: |
      Synthesize all the information gathered so far:
      {{aggregated_inputs}}
      
      Provide a final comprehensive assessment.
    output_key: "final_answer"
    branches:
      - condition:
          type: "always"
        target_node: "final_summary"
  
  final_summary:
    node_id: "final_summary"
    prompt: "Provide a comprehensive summary of the urban planning characteristics of this area."
    node_type: "leaf"
    output_key: "final_answer"
```

**Many-to-One Pattern Example**:
```yaml
# Example: Multiple analysis paths converge at a single synthesis node
nodes:
  # Multiple parallel analysis paths
  structural_analysis:
    node_id: "structural_analysis"
    prompt: "Analyze the structural elements in this image."
    node_type: "question"
    output_key: "structural_analysis"
    branches:
      - condition:
          type: "always"
        target_node: "synthesis"  # All paths lead to synthesis
  
  functional_analysis:
    node_id: "functional_analysis"
    prompt: "Analyze the functional aspects of this urban space."
    node_type: "question"
    output_key: "functional_analysis"
    branches:
      - condition:
          type: "always"
        target_node: "synthesis"  # Converges at synthesis
  
  aesthetic_analysis:
    node_id: "aesthetic_analysis"
    prompt: "Describe the aesthetic and visual qualities."
    node_type: "question"
    output_key: "aesthetic_analysis"
    branches:
      - condition:
          type: "always"
        target_node: "synthesis"  # Converges at synthesis
  
  # Convergence node: receives all three analyses
  synthesis:
    node_id: "synthesis"
    node_type: "convergence"
    aggregation_strategy: "concatenate"  # or "summarize", "list", "json"
    prompt: |
      Based on the following comprehensive analyses:
      
      Structural Analysis: {{structural_analysis}}
      Functional Analysis: {{functional_analysis}}
      Aesthetic Analysis: {{aesthetic_analysis}}
      
      Provide a holistic assessment integrating all perspectives.
    output_key: "final_assessment"
    depends_on: ["structural_analysis", "functional_analysis", "aesthetic_analysis"]
    branches:
      - condition:
          type: "always"
        target_node: "final_answer"
  
  final_answer:
    node_id: "final_answer"
    prompt: "Provide the final answer based on your comprehensive assessment."
    node_type: "leaf"
    output_key: "final_answer"
```

#### 6.3.2 Decision Tree Loading and Execution

**Implementation**:

```python
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json
import yaml

@dataclass
class TreeNode:
    """Represents a node in a decision tree."""
    node_id: str
    prompt: str
    node_type: str  # "question", "decision", "leaf", "action"
    output_key: Optional[str] = None
    branches: List[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None

class DecisionTree:
    """Manages decision tree structure and navigation."""
    
    def __init__(self, tree_config: Dict[str, Any]):
        self.tree_id = tree_config.get("tree_id")
        self.version = tree_config.get("version", "1.0.0")
        self.root_node_id = tree_config.get("root_node")
        self.nodes = {}
        
        # Load nodes
        for node_id, node_data in tree_config.get("nodes", {}).items():
            self.nodes[node_id] = TreeNode(
                node_id=node_data["node_id"],
                prompt=node_data["prompt"],
                node_type=node_data.get("node_type", "question"),
                output_key=node_data.get("output_key"),
                branches=node_data.get("branches", []),
                metadata=node_data.get("metadata", {})
            )
    
    @classmethod
    def from_json(cls, json_path: str) -> "DecisionTree":
        """Load decision tree from JSON file."""
        with open(json_path, "r") as f:
            config = json.load(f)
        return cls(config)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DecisionTree":
        """Load decision tree from YAML file."""
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(config)
    
    def get_next_node(self, current_node_id: str, model_response: str, 
                     context: Dict[str, Any] = None) -> Optional[str]:
        """Determine next node based on current node and model response."""
        current_node = self.nodes.get(current_node_id)
        if not current_node or not current_node.branches:
            return None
        
        # Evaluate branches in order of weight (highest first)
        sorted_branches = sorted(
            current_node.branches,
            key=lambda b: b.get("weight", 0.0),
            reverse=True
        )
        
        for branch in sorted_branches:
            if self._evaluate_condition(branch.get("condition", {}), 
                                       model_response, context):
                return branch["target_node"]
        
        return None
    
    def _evaluate_condition(self, condition: Dict[str, Any], 
                           response: str, context: Dict[str, Any] = None) -> bool:
        """Evaluate a branch condition."""
        cond_type = condition.get("type", "default")
        
        if cond_type == "keyword_match":
            keywords = condition.get("keywords", [])
            response_lower = response.lower()
            return any(kw.lower() in response_lower for kw in keywords)
        
        elif cond_type == "confidence_threshold":
            threshold = condition.get("threshold", 0.5)
            confidence = context.get("confidence", 0.0) if context else 0.0
            return confidence >= threshold
        
        elif cond_type == "regex_match":
            pattern = condition.get("pattern", "")
            import re
            return bool(re.search(pattern, response, re.IGNORECASE))
        
        elif cond_type == "always":
            return True
        
        elif cond_type == "default":
            return True  # Default branch
        
        return False
    
    def get_convergence_inputs(self, node_id: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Get inputs from all predecessor nodes that feed into a convergence node."""
        node = self.nodes.get(node_id)
        if not node or node.node_type != "convergence":
            return {}
        
        # Find all nodes that have this node as a target
        predecessor_inputs = {}
        for pred_node_id, pred_node in self.nodes.items():
            if pred_node.branches:
                for branch in pred_node.branches:
                    if branch.get("target_node") == node_id:
                        # Get output from this predecessor
                        if pred_node.output_key and pred_node.output_key in results:
                            predecessor_inputs[pred_node.output_key] = results[pred_node.output_key]
        
        # Also check depends_on metadata if present
        if node.metadata and node.metadata.get("depends_on"):
            for dep_key in node.metadata["depends_on"]:
                if dep_key in results:
                    predecessor_inputs[dep_key] = results[dep_key]
        
        return predecessor_inputs
    
    def aggregate_inputs(self, inputs: Dict[str, Any], strategy: str = "concatenate") -> str:
        """Aggregate multiple inputs according to strategy."""
        if not inputs:
            return ""
        
        if strategy == "concatenate":
            return "\n\n".join([f"{k}: {v}" for k, v in inputs.items()])
        elif strategy == "summarize":
            # Return formatted list for summarization
            return "\n".join([f"- {k}: {v}" for k, v in inputs.items()])
        elif strategy == "list":
            return "\n".join([f"- {v}" for v in inputs.values()])
        elif strategy == "json":
            import json
            return json.dumps(inputs, indent=2)
        else:
            # Default to concatenate
            return "\n\n".join([f"{k}: {v}" for k, v in inputs.items()])
    
    def _find_predecessors(self, node_id: str) -> List[str]:
        """Find all predecessor nodes that lead to a given node."""
        predecessors = []
        for pred_node_id, pred_node in self.nodes.items():
            if pred_node.branches:
                for branch in pred_node.branches:
                    if branch.get("target_node") == node_id:
                        predecessors.append(pred_node_id)
        return predecessors
    
    def traverse(self, initial_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Traverse the tree and collect prompts/responses."""
        results = []
        current_node_id = self.root_node_id
        context = initial_context or {}
        
        visited = set()  # Prevent infinite loops
        max_depth = 10  # Safety limit
        
        depth = 0
        while current_node_id and depth < max_depth:
            if current_node_id in visited:
                break  # Cycle detected
            
            visited.add(current_node_id)
            current_node = self.nodes.get(current_node_id)
            
            if not current_node:
                break
            
            results.append({
                "node_id": current_node_id,
                "prompt": current_node.prompt,
                "output_key": current_node.output_key,
                "node_type": current_node.node_type
            })
            
            # If leaf node, we're done
            if current_node.node_type == "leaf":
                break
            
            # Get model response (would be called from VQA stage)
            # For now, we'll need to integrate this with the actual VQA execution
            # This is a placeholder - actual implementation would call the model
            
            depth += 1
        
        return results
```

#### 6.3.3 Integration with VQA Stage and Ray Data

**Configuration** (Hydra-compatible):
```yaml
prompt:
  decision_tree:
    enabled: true
    tree_path: "dagspaces/urbanvqa/conf/prompts/decision_trees/urban_planning_vqa.yaml"
    tree_format: "yaml"  # or "json"
    max_depth: 10  # Maximum tree depth
    enable_cycle_detection: true
```

**VQA Stage Integration** (vLLM-compatible message format with many-to-one support):
```python
def run_vqa_with_decision_tree(row: Dict[str, Any], cfg: DictConfig) -> Dict[str, Any]:
    """Run VQA with decision tree prompting using vLLM-compatible format."""
    from dagspaces.urbanvqa.prompts.decision_tree import DecisionTree
    
    # Load decision tree
    tree_config = cfg.prompt.decision_tree
    if tree_config.tree_format == "yaml":
        tree = DecisionTree.from_yaml(tree_config.tree_path)
    else:
        tree = DecisionTree.from_json(tree_config.tree_path)
    
    # Traverse tree
    results = {}
    current_node_id = tree.root_node_id
    image = row.get("image")  # PIL Image or image path
    sample_id = row.get("sample_id", "unknown")
    
    # Convert image to appropriate format for vLLM
    # Best Practice: Use OpenAI chat format with proper content types
    if isinstance(image, str):
        # Image URL (http/https)
        if image.startswith(("http://", "https://")):
            image_content = {
                "type": "image_url",
                "image_url": {"url": image}
            }
        # Base64 encoded string
        elif image.startswith("data:image/"):
            image_content = {
                "type": "image_url",
                "image_url": {"url": image}
            }
        else:
            # Local path - load as PIL Image
            from PIL import Image as PILImage
            image = PILImage.open(image).convert("RGB")
            image_content = {"type": "image", "image": image}
    else:
        # PIL Image object - use image format
        image_content = {"type": "image", "image": image}
    
    visited_nodes = set()
    depth = 0
    max_depth = tree_config.get("max_depth", 10)
    convergence_nodes_visited = {}  # Track convergence nodes and their inputs
    
    while current_node_id and depth < max_depth:
        if tree_config.enable_cycle_detection and current_node_id in visited_nodes:
            break  # Cycle detected
        
        visited_nodes.add(current_node_id)
        node = tree.nodes[current_node_id]
        
        # Handle convergence nodes (many-to-one)
        if node.node_type == "convergence":
            # Get inputs from all predecessor nodes
            convergence_inputs = tree.get_convergence_inputs(current_node_id, results)
            
            # Aggregate inputs according to strategy
            aggregation_strategy = node.metadata.get("aggregation_strategy", "concatenate") if node.metadata else "concatenate"
            aggregated_text = tree.aggregate_inputs(convergence_inputs, aggregation_strategy)
            
            # Use aggregation prompt if provided, otherwise use node prompt
            if node.metadata and node.metadata.get("aggregation_prompt"):
                prompt_template = node.metadata["aggregation_prompt"]
                # Replace placeholders in aggregation prompt
                for key, value in convergence_inputs.items():
                    prompt_template = prompt_template.replace(f"{{{{{key}}}}}", str(value))
                node_prompt = prompt_template
            else:
                # Use node prompt with {{aggregated_inputs}} placeholder
                node_prompt = node.prompt.replace("{{aggregated_inputs}}", aggregated_text)
                # Also replace individual input placeholders
                for key, value in convergence_inputs.items():
                    node_prompt = node_prompt.replace(f"{{{{{key}}}}}", str(value))
            
            # Track convergence for postprocessing
            convergence_nodes_visited[current_node_id] = {
                "inputs": convergence_inputs,
                "aggregated": aggregated_text
            }
        else:
            # Regular node - use standard prompt
            node_prompt = node.prompt
            # Replace placeholders with previous results
            for key, value in results.items():
                node_prompt = node_prompt.replace(f"{{{{{key}}}}}", str(value))
            
            # Replace user question placeholder
            if "{{final_question}}" in node_prompt or "{{user_question}}" in node_prompt:
                node_prompt = node_prompt.replace("{{final_question}}", row.get("prompt", ""))
                node_prompt = node_prompt.replace("{{user_question}}", row.get("prompt", ""))
        
        # Build vLLM-compatible messages (OpenAI chat format)
        messages = [
            {"role": "system", "content": cfg.prompt.system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": node_prompt},
                    image_content  # Reuse same image for all nodes
                ]
            }
        ]
        
        # Call vLLM model (via Ray Data LLM processor)
        # Best Practice: In Ray Data context, this is handled by build_llm_processor
        # The processor automatically calls vLLM with proper message format
        # For standalone use, would use: llm.generate({"prompt": messages, "multi_modal_data": {"image": image}})
        # But in Ray Data, processor handles this automatically
        response = call_vllm_model(messages, cfg)  # Placeholder - actual call handled by processor
        
        # Store response
        if node.output_key:
            results[node.output_key] = response
        
        # Get next node based on response
        current_node_id = tree.get_next_node(
            current_node_id,
            response,
            context={"confidence": extract_confidence(response) if hasattr(cfg, "confidence_extraction") else None}
        )
        
        if node.node_type == "leaf":
            break
        
        depth += 1
    
    return {
        "sample_id": sample_id,
        **results,
        "metadata": {
            "tree_id": tree.tree_id,
            "version": tree.version,
            "nodes_visited": list(visited_nodes),
            "depth": depth,
            "convergence_nodes": convergence_nodes_visited
        }
    }
```

**Ray Data Integration** (Batch Processing with Many-to-One Support):
```python
def run_vqa_with_decision_tree_batch(batch: Dict[str, Any], cfg: DictConfig) -> Dict[str, Any]:
    """Run VQA with decision tree prompting in batch mode via Ray Data."""
    from dagspaces.urbanvqa.prompts.decision_tree import DecisionTree
    
    # Load decision tree (shared across batch)
    tree_config = cfg.prompt.decision_tree
    tree = DecisionTree.from_yaml(tree_config.tree_path)
    
    # Process each row in batch
    results = []
    for idx in range(len(batch["sample_id"])):
        row = {k: v[idx] if isinstance(v, list) else v for k, v in batch.items()}
        result = run_vqa_with_decision_tree(row, cfg)
        results.append(result)
    
    # Aggregate results back into batch format
    output_batch = {}
    for key in results[0].keys():
        output_batch[key] = [r[key] for r in results]
    
    return output_batch

def process_parallel_convergence(group: List[Dict[str, Any]], cfg: DictConfig) -> Dict[str, Any]:
    """Process a group of parallel nodes that converge at a single node."""
    # In Ray Data, we can use map_batches to process parallel groups
    # Each parallel step is processed independently, then aggregated
    
    parallel_results = {}
    for step_config in group:
        step_name = step_config["name"]
        # Process each parallel step (would call vLLM)
        # For now, return placeholder
        parallel_results[step_config["output_key"]] = f"Result from {step_name}"
    
    # Aggregate parallel results for convergence node
    return parallel_results
```

**Note on Stateful vs Stateless Processors**:
- **build_llm_processor**: Best for stateless preprocessing. For stateful operations (like loading decision trees), use class-based processors with `map_batches` instead, or ensure state is properly initialized per worker.
- **Class-based with map_batches**: For fully stateful operations where you need explicit control over worker initialization, use `map_batches` with a class that implements `__init__` and `__call__`.
- **build_llm_processor with closures**: Can capture state via closures, but for complex stateful operations, prefer `map_batches`.
```python
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig

class DecisionTreeProcessor:
    """Stateful processor class for decision tree traversal.
    
    Best Practice: Load decision tree once in __init__ per worker,
    not on every preprocessing call. This avoids repeated file I/O.
    """
    def __init__(self, tree_path: str, tree_format: str, cfg: DictConfig):
        from dagspaces.urbanvqa.prompts.decision_tree import DecisionTree
        # Load decision tree once per worker
        if tree_format == "yaml":
            self.tree = DecisionTree.from_yaml(tree_path)
        else:
            self.tree = DecisionTree.from_json(tree_path)
        self.cfg = cfg
    
    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess row to traverse decision tree and build messages."""
        if not self.cfg.prompt.decision_tree.enabled:
            # Fallback to simple prompt
            # Build messages with image for VQA
            image_content = _prepare_image_content(row.get("image"), row.get("sample_id"))
            return {
                "messages": [
                    {"role": "system", "content": self.cfg.prompt.system},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": row["prompt"]},
                            image_content
                        ]
                    }
                ],
                "sampling_params": self.cfg.sampling_params_vqa
            }
        
        # Check if we're at a convergence node
        current_step = row.get("_tree_current_step")
        if current_step and current_step.get("node_type") == "convergence":
            # Handle many-to-one aggregation
            convergence_inputs = self.tree.get_convergence_inputs(
                current_step["node_id"],
                row.get("_tree_results", {})
            )
            
            # Aggregate inputs
            strategy = current_step.get("aggregation_strategy", "concatenate")
            aggregated = self.tree.aggregate_inputs(convergence_inputs, strategy)
            
            # Build prompt with aggregated inputs
            prompt_template = current_step.get("aggregation_prompt") or current_step["prompt"]
            prompt = prompt_template.replace("{{aggregated_inputs}}", aggregated)
            for key, value in convergence_inputs.items():
                prompt = prompt.replace(f"{{{{{key}}}}}", str(value))
        else:
            # Regular node processing
            prompt = current_step["prompt"] if current_step else row["prompt"]
        
        # Build messages in vLLM format
        image_content = _prepare_image_content(row.get("image"), row.get("sample_id"))
        messages = [
            {"role": "system", "content": self.cfg.prompt.system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    image_content
                ]
            }
        ]
        
        return {
            "messages": messages,
            "sampling_params": self.cfg.sampling_params_vqa,
            "_tree_step": current_step  # Track for postprocessing
        }
    
    def postprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Extract final answer from decision tree outputs."""
        return {
            "sample_id": row.get("sample_id"),
            "answer": row.get("generated_text"),  # Final answer from leaf node
            "metadata": row.get("metadata", {})
        }

def create_decision_tree_vqa_processor(cfg: DictConfig):
    """Create Ray Data LLM processor with decision tree support (including many-to-one)."""
    
    processor_config = vLLMEngineProcessorConfig(
        model_source=cfg.model.model_source,
        engine_kwargs={
            **cfg.model.engine_kwargs,
            # Best Practice: Always include multimodal settings
            "limit_mm_per_prompt": cfg.model.engine_kwargs.get("limit_mm_per_prompt", {"image": 1}),
            "trust_remote_code": cfg.model.engine_kwargs.get("trust_remote_code", True),  # Required for custom vision models
        },
        concurrency=cfg.model.concurrency,
        batch_size=cfg.model.batch_size,
        has_image=True,  # Best Practice: Enable multimodal for VQA
    )
    
    # Best Practice: For stateful processors with build_llm_processor,
    # Ray Data automatically creates one processor instance per worker.
    # The preprocess/postprocess functions are called on that instance.
    # DecisionTreeProcessor loads the tree once in __init__ per worker.
    # Note: build_llm_processor accepts callable functions, not class instances.
    # For stateful operations, we use a closure pattern or class methods.
    
    processor_instance = DecisionTreeProcessor(
        tree_path=cfg.prompt.decision_tree.tree_path,
        tree_format=cfg.prompt.decision_tree.tree_format,
        cfg=cfg
    )
    
    processor = build_llm_processor(
        processor_config,
        # Best Practice: Pass bound methods - Ray will handle worker instantiation
        # Ray Data creates one DecisionTreeProcessor instance per worker automatically
        preprocess=processor_instance.preprocess,
        postprocess=processor_instance.postprocess
    )
    
    return processor
```

#### 6.3.4 Decision Tree Configuration Files

**New Directory**: `dagspaces/urbanvqa/conf/prompts/decision_trees/`

**Example Files**:
- `urban_planning_vqa.yaml` - Urban planning decision tree
- `building_classification.yaml` - Building type classification tree
- `infrastructure_analysis.yaml` - Infrastructure analysis tree

**File Structure**:
```
dagspaces/urbanvqa/conf/prompts/decision_trees/
├── README.md  # Documentation on creating decision trees
├── schema.json  # JSON schema for validation
├── urban_planning_vqa.yaml
├── building_classification.yaml
└── infrastructure_analysis.yaml
```

### 6.4 Other Dynamic Prompting Techniques

**Important**: All techniques must use vLLM-compatible message formats and integrate with Ray Data's `build_llm_processor` architecture.

#### 6.4.1 Chain-of-Thought (CoT) Prompting

**Description**: Encourages the model to generate intermediate reasoning steps before providing the final answer, improving handling of complex multi-step problems.

**vLLM/Ray Data Compatibility**: CoT prompts are formatted as standard OpenAI chat messages. The reasoning structure is embedded in the prompt text itself.

**Implementation** (Hydra-configurable):
```yaml
prompt:
  chain_of_thought:
    enabled: true
    template: |
      Let's solve this step by step:
      
      1. First, analyze the image: {{image_analysis}}
      2. Then, reason about the question: {{reasoning}}
      3. Finally, provide the answer: {{answer}}
```

**Ray Data Integration**:
```python
def preprocess_cot(row: Dict[str, Any], cfg: DictConfig) -> Dict[str, Any]:
    """Preprocess with Chain-of-Thought structure."""
    if not cfg.prompt.chain_of_thought.enabled:
        # Best Practice: Always include image for VQA, even in fallback cases
        image_content = _prepare_image_content(row.get("image"), row.get("sample_id"))
        return {
            "messages": [
                {"role": "system", "content": cfg.prompt.system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": row["prompt"]},
                        image_content
                    ]
                }
            ],
            "sampling_params": cfg.sampling_params_vqa
        }
    
    # Build CoT prompt
    cot_prompt = cfg.prompt.chain_of_thought.template.replace(
        "{{image_analysis}}", "Analyze the image carefully."
    ).replace("{{reasoning}}", "Reason about the question.").replace(
        "{{answer}}", "Provide your final answer."
    )
    
    # Build vLLM-compatible messages
    image_content = _prepare_image_content(row["image"], row.get("sample_id"))
    messages = [
        {"role": "system", "content": cfg.prompt.system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": cot_prompt},
                image_content
            ]
        }
    ]
    
    return {
        "messages": messages,
        "sampling_params": cfg.sampling_params_vqa
    }
```

**VQA Integration**: Automatically inject CoT structure into prompts for complex questions via Ray Data preprocessing.

#### 6.4.2 ReAct (Reasoning and Acting) Pattern

**Description**: Combines reasoning and action steps, allowing the model to interact with external tools or data sources dynamically.

**vLLM Compatibility**: ReAct uses function calling/tool use features if supported by the model. Otherwise, simulate via structured prompts.

**Implementation** (Hydra-configurable):
```yaml
prompt:
  react:
    enabled: true
    tools:
      - name: "lookup_building_code"
        description: "Look up building code information"
        function: "lookup_building_code"
      - name: "check_urban_planning_regs"
        description: "Check urban planning regulations"
        function: "check_urban_planning_regs"
```

**Ray Data Integration**:
```python
def preprocess_react(row: Dict[str, Any], cfg: DictConfig) -> Dict[str, Any]:
    """Preprocess with ReAct pattern."""
    if not cfg.prompt.react.enabled:
        return preprocess_simple(row, cfg)
    
    # Build ReAct prompt with tool descriptions
    tool_descriptions = "\n".join([
        f"- {tool.name}: {tool.description}"
        for tool in cfg.prompt.react.tools
    ])
    
    react_prompt = f"""You have access to the following tools:
{tool_descriptions}

Use these tools to answer the question. Format your reasoning as:
Thought: [your reasoning]
Action: [tool name]
Action Input: [parameters]
Observation: [tool result]
... (repeat as needed)
Final Answer: [final answer]"""
    
    image_content = _prepare_image_content(row["image"], row.get("sample_id"))
    messages = [
        {"role": "system", "content": cfg.prompt.system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{react_prompt}\n\nQuestion: {row['prompt']}"},
                image_content
            ]
        }
    ]
    
    return {
        "messages": messages,
        "sampling_params": cfg.sampling_params_vqa
    }
```

**Use Case**: For VQA tasks requiring external knowledge (e.g., checking building codes, regulations).

#### 6.4.3 Self-Consistency Prompting

**Description**: Generates multiple reasoning paths and selects the most consistent answer, enhancing reliability for ambiguous or complex tasks.

**vLLM/Ray Data Compatibility**: Run multiple inference passes with slight prompt variations, then aggregate results using Ray Data's batch processing capabilities.

**Implementation** (Hydra-configurable):
```yaml
prompt:
  self_consistency:
    enabled: true
    num_samples: 5  # Generate 5 different reasoning paths
    consistency_threshold: 0.7  # Minimum agreement threshold
    voting_method: "majority"  # or "weighted"
    prompt_variations:
      - "Think step by step: {prompt}"
      - "Analyze carefully: {prompt}"
      - "Consider multiple perspectives: {prompt}"
```

**Ray Data Integration**:
```python
def preprocess_self_consistency(row: Dict[str, Any], cfg: DictConfig) -> Dict[str, Any]:
    """Preprocess for self-consistency prompting."""
    if not cfg.prompt.self_consistency.enabled:
        return preprocess_simple(row, cfg)
    
    # Create multiple prompt variations
    base_prompt = row["prompt"]
    variations = cfg.prompt.self_consistency.prompt_variations
    
    # This will be handled by expanding the dataset
    # For now, use first variation
    varied_prompt = variations[0].format(prompt=base_prompt)
    
    image_content = _prepare_image_content(row["image"], row.get("sample_id"))
    messages = [
        {"role": "system", "content": cfg.prompt.system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": varied_prompt},
                image_content
            ]
        }
    ]
    
    return {
        "messages": messages,
        "sampling_params": cfg.sampling_params_vqa,
        "_consistency_sample_id": row.get("sample_id")  # For grouping
    }

# Postprocess aggregates multiple samples
def postprocess_self_consistency(row: Dict[str, Any], cfg: DictConfig) -> Dict[str, Any]:
    """Aggregate self-consistency results."""
    # Group by sample_id and aggregate responses
    # This would be done via Ray Data groupby or custom aggregation
    return {
        "sample_id": row.get("sample_id"),
        "answer": row.get("generated_text"),
        "consistency_score": calculate_consistency(row.get("generated_text"))
    }
```

**VQA Integration**: Use Ray Data's `map_batches` with expanded dataset (duplicate rows with variations), then aggregate results.

#### 6.4.4 Retrieval-Augmented Prompting (RAP)

**Description**: Enhances prompts with relevant information retrieved from external sources, improving accuracy and grounding.

**vLLM/Ray Data Compatibility**: RAP augments prompt text before sending to vLLM. Retrieval happens in preprocessing stage.

**Implementation** (Hydra-configurable):
```yaml
prompt:
  retrieval_augmented:
    enabled: true
    knowledge_base: "urban_planning_kb"  # Path to knowledge base
    max_context_tokens: 500
    retrieval_method: "semantic_search"  # or "keyword_match", "hybrid"
    top_k: 3  # Number of retrieved documents
```

**Ray Data Integration**:
```python
def preprocess_rap(row: Dict[str, Any], cfg: DictConfig) -> Dict[str, Any]:
    """Preprocess with retrieval-augmented prompting."""
    if not cfg.prompt.retrieval_augmented.enabled:
        return preprocess_simple(row, cfg)
    
    # Retrieve relevant context
    kb = load_knowledge_base(cfg.prompt.retrieval_augmented.knowledge_base)
    retrieved = retrieve_context(
        query=row["prompt"],
        kb=kb,
        method=cfg.prompt.retrieval_augmented.retrieval_method,
        top_k=cfg.prompt.retrieval_augmented.top_k
    )
    
    # Build augmented prompt
    context_text = "\n\n".join([doc["text"] for doc in retrieved])
    augmented_prompt = f"""Use the following context to answer the question:

Context:
{context_text}

Question: {row["prompt"]}"""
    
    image_content = _prepare_image_content(row["image"], row.get("sample_id"))
    messages = [
        {"role": "system", "content": cfg.prompt.system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": augmented_prompt},
                image_content
            ]
        }
    ]
    
    return {
        "messages": messages,
        "sampling_params": cfg.sampling_params_vqa
    }
```

**Use Case**: Provide relevant urban planning context, building codes, or historical data alongside the image.

#### 6.4.5 Prompt Chaining

**Description**: Breaks down complex tasks into a sequence of smaller, interconnected prompts, guiding the model through a structured multi-step reasoning process.

**vLLM/Ray Data Compatibility**: Each chain step is a separate vLLM call. State is maintained across steps via Ray Data map functions.

**Implementation** (Hydra-configurable):
```yaml
prompt:
  chaining:
    enabled: true
    chain:
      - step: 1
        prompt: "Identify all visible structures in this image."
        output_key: "structures"
      - step: 2
        prompt: "For each structure in {{structures}}, classify its type."
        output_key: "structure_types"
      - step: 3
        prompt: "Based on {{structure_types}}, answer: {{user_question}}"
        output_key: "final_answer"
```

**Ray Data Integration**:
```python
def preprocess_chaining(row: Dict[str, Any], cfg: DictConfig) -> Dict[str, Any]:
    """Preprocess for prompt chaining."""
    if not cfg.prompt.chaining.enabled:
        return preprocess_simple(row, cfg)
    
    # Execute chain steps sequentially
    chain_state = {}
    chain_state["user_question"] = row["prompt"]
    
    for chain_step in cfg.prompt.chaining.chain:
        # Format prompt with state
        step_prompt = chain_step.prompt
        for key, value in chain_state.items():
            step_prompt = step_prompt.replace(f"{{{{{key}}}}}}", str(value))
        
        image_content = _prepare_image_content(row["image"], row.get("sample_id"))
        messages = [
            {"role": "system", "content": cfg.prompt.system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": step_prompt},
                    image_content
                ]
            }
        ]
        
        # Best Practice: In Ray Data context, vLLM calls are handled by processor
        # This is a placeholder - actual implementation uses build_llm_processor
        response = call_vllm_model(messages, cfg)  # Placeholder - handled by processor
        
        # Store result in state
        chain_state[chain_step.output_key] = response
        
        # If this is the final step, return it
        if chain_step == cfg.prompt.chaining.chain[-1]:
            return {
                "messages": messages,  # Last step's messages
                "sampling_params": cfg.sampling_params_vqa,
                "_chain_state": chain_state  # For postprocessing
            }
    
    return {
        "messages": messages,
        "sampling_params": cfg.sampling_params_vqa
    }
```

**Relation to Hierarchical Prompts**: Similar concept but more flexible branching logic. Can be implemented as sequential Ray Data map operations.

#### 6.4.6 Contextual Dynamic Prompting

**Description**: Adjusts prompts in real-time based on conversation context, user intent, or intermediate model outputs.

**vLLM/Ray Data Compatibility**: Context adaptation happens in preprocessing stage. State is maintained via Ray Data stateful actors or metadata columns.

**Implementation** (Hydra-configurable):
```yaml
prompt:
  contextual:
    enabled: true
    context_window: 5  # Number of previous interactions to consider
    adaptation_rules:
      - condition: "low_confidence"
        action: "add_detail_prompt"
        detail_prompt: "Please provide more specific details."
      - condition: "ambiguous_answer"
        action: "add_clarification"
        clarification: "Can you clarify which aspect you're referring to?"
```

**Ray Data Integration**:
```python
# Use Ray Data stateful actors for context tracking
@ray.remote
class ContextTracker:
    """Ray actor for managing context history across batches.
    
    Best Practice: Use Ray actors for shared state, not global variables.
    Actors maintain state across method calls and are accessible from
    multiple workers.
    """
    def __init__(self):
        self.context_history = {}
    
    def update(self, sample_id: str, response: str, confidence: float):
        if sample_id not in self.context_history:
            self.context_history[sample_id] = []
        self.context_history[sample_id].append({
            "response": response,
            "confidence": confidence
        })
    
    def get_context(self, sample_id: str):
        return self.context_history.get(sample_id, [])

# Initialize actor (typically done once at pipeline start)
# Best Practice: Use named actors for easy retrieval across workers
try:
    tracker = ray.get_actor("context_tracker")
except ValueError:
    # Actor doesn't exist yet, create it
    tracker = ContextTracker.options(name="context_tracker").remote()

def preprocess_contextual(row: Dict[str, Any], cfg: DictConfig) -> Dict[str, Any]:
    """Preprocess with contextual adaptation."""
    if not cfg.prompt.contextual.enabled:
        return preprocess_simple(row, cfg)
    
    # Get context history (with error handling)
    try:
        tracker = ray.get_actor("context_tracker")
        context = ray.get(tracker.get_context.remote(row.get("sample_id")))
    except (ValueError, ray.exceptions.GetTimeoutError):
        # Actor not available or timeout - use empty context
        context = []
    
    # Adapt prompt based on context
    base_prompt = row["prompt"]
    adapted_prompt = base_prompt
    
    if context:
        # Check adaptation rules
        last_response = context[-1] if context else None
        if last_response and last_response.get("confidence", 1.0) < 0.5:
            # Low confidence - add detail prompt
            adapted_prompt = f"{base_prompt}\n\nPlease provide more specific details."
    
    image_content = _prepare_image_content(row["image"], row.get("sample_id"))
    messages = [
        {"role": "system", "content": cfg.prompt.system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": adapted_prompt},
                image_content
            ]
        }
    ]
    
    return {
        "messages": messages,
        "sampling_params": cfg.sampling_params_vqa
    }
```

**VQA Integration**: Monitor model confidence and adapt follow-up prompts accordingly using Ray actors for state management.

#### 6.4.7 Adaptive Prompting

**Description**: Modifies prompt structures and content based on task complexity and user input to enhance reasoning capabilities.

**vLLM/Ray Data Compatibility**: Complexity assessment happens in preprocessing. Different prompt variants are selected based on assessment.

**Implementation** (Hydra-configurable):
```yaml
prompt:
  adaptive:
    enabled: true
    complexity_assessment:
      method: "token_count"  # or "semantic_complexity", "heuristic"
      thresholds:
        simple: 50  # tokens
        medium: 150
        complex: 300
    prompt_variants:
      simple:
        prompt: "{{user_question}}"
      medium:
        prompt: "Analyze the image and answer: {{user_question}}"
      complex:
        prompt: |
          Let's break down this complex question:
          1. First, describe what you see: {{image_description}}
          2. Then, analyze the context: {{context_analysis}}
          3. Finally, answer: {{user_question}}
```

**Ray Data Integration**:
```python
def assess_complexity(prompt: str, method: str) -> str:
    """Assess prompt complexity."""
    if method == "token_count":
        # Simple token count heuristic
        token_count = len(prompt.split())
        if token_count < 50:
            return "simple"
        elif token_count < 150:
            return "medium"
        else:
            return "complex"
    elif method == "semantic_complexity":
        # Use embedding similarity or other semantic methods
        # Placeholder
        return "medium"
    return "medium"

def preprocess_adaptive(row: Dict[str, Any], cfg: DictConfig) -> Dict[str, Any]:
    """Preprocess with adaptive prompting."""
    if not cfg.prompt.adaptive.enabled:
        return preprocess_simple(row, cfg)
    
    # Assess complexity
    complexity = assess_complexity(
        row["prompt"],
        cfg.prompt.adaptive.complexity_assessment.method
    )
    
    # Select prompt variant
    variant = cfg.prompt.adaptive.prompt_variants[complexity]
    adapted_prompt = variant.prompt.replace("{{user_question}}", row["prompt"])
    
    image_content = _prepare_image_content(row["image"], row.get("sample_id"))
    messages = [
        {"role": "system", "content": cfg.prompt.system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": adapted_prompt},
                image_content
            ]
        }
    ]
    
    return {
        "messages": messages,
        "sampling_params": cfg.sampling_params_vqa,
        "_complexity": complexity  # For monitoring
    }
```

**VQA Integration**: Automatically adjust prompt complexity based on question length or semantic analysis in preprocessing stage.

### 6.5 Dynamic Prompting Framework Architecture

**Unified Configuration** (Hydra-compatible):
```yaml
prompt:
  # Base prompt
  system: "You are a helpful assistant that answers questions about images."
  user_template: "{{prompt}}"
  
  # Enable/disable techniques (Hydra config groups)
  dynamic_prompts: true
  hierarchical:
    enabled: false
    # ... hierarchical config
  decision_tree:
    enabled: false
    # ... decision tree config
  chain_of_thought:
    enabled: false
    # ... CoT config
  react:
    enabled: false
    # ... ReAct config
  self_consistency:
    enabled: false
    # ... self-consistency config
  retrieval_augmented:
    enabled: false
    # ... RAP config
  chaining:
    enabled: false
    # ... chaining config
  contextual:
    enabled: false
    # ... contextual config
  adaptive:
    enabled: false
    # ... adaptive config
  
  # Priority order (if multiple enabled)
  priority: ["decision_tree", "hierarchical", "chain_of_thought", "adaptive"]
```

**Execution Order** (Integrated with Ray Data pipeline):
1. **Pre-processing**: Adaptive prompting (adjust complexity) → RAP (augment with external knowledge)
2. **Structure**: Decision tree or hierarchical prompts
3. **Reasoning**: Chain-of-thought or ReAct
4. **Validation**: Self-consistency checking (if enabled)
5. **Post-processing**: Contextual adaptation based on results

**Ray Data Processor Integration**:
```python
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig
from omegaconf import DictConfig

def create_unified_vqa_processor(cfg: DictConfig):
    """Create unified Ray Data LLM processor supporting all dynamic prompting techniques."""
    
        processor_config = vLLMEngineProcessorConfig(
        model_source=cfg.model.model_source,
        engine_kwargs={
            **cfg.model.engine_kwargs,
            # Best Practice: Always include multimodal settings
            "limit_mm_per_prompt": cfg.model.engine_kwargs.get("limit_mm_per_prompt", {"image": 1}),
            "trust_remote_code": cfg.model.engine_kwargs.get("trust_remote_code", True),  # Required for custom vision models
        },
        concurrency=cfg.model.concurrency,  # Best Practice: Set to number of model replicas
        # Note: When using tensor parallelism, concurrency = total_gpus / tensor_parallel_size
        # Example: 2 GPUs with tensor_parallel_size=2 → concurrency=1 (one replica using both GPUs)
        batch_size=cfg.model.batch_size,  # Best Practice: 16-64 for LLMs, adjust based on GPU memory
        has_image=True,  # Best Practice: Always enable multimodal for VQA
        accelerator_type=cfg.model.get("accelerator_type"),
    )
    
    def unified_preprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        """Unified preprocessing that applies enabled techniques in priority order."""
        enabled_techniques = []
        
        # Determine which techniques are enabled
        if cfg.prompt.decision_tree.enabled:
            enabled_techniques.append(("decision_tree", 1))
        if cfg.prompt.hierarchical.enabled:
            enabled_techniques.append(("hierarchical", 2))
        if cfg.prompt.chain_of_thought.enabled:
            enabled_techniques.append(("chain_of_thought", 3))
        if cfg.prompt.adaptive.enabled:
            enabled_techniques.append(("adaptive", 0))  # Pre-processing
        
        # Sort by priority
        enabled_techniques.sort(key=lambda x: x[1])
        
        # Apply techniques in order
        current_row = dict(row)
        
        # 1. Pre-processing techniques (adaptive, RAP)
        if cfg.prompt.adaptive.enabled:
            current_row = preprocess_adaptive(current_row, cfg)
        if cfg.prompt.retrieval_augmented.enabled:
            current_row = preprocess_rap(current_row, cfg)
        
        # 2. Structural techniques (decision tree, hierarchical)
        if cfg.prompt.decision_tree.enabled:
            # Decision tree handles its own traversal
            return preprocess_decision_tree(current_row, cfg)
        elif cfg.prompt.hierarchical.enabled:
            return preprocess_hierarchical(current_row, cfg)
        
        # 3. Reasoning techniques (CoT, ReAct, chaining)
        if cfg.prompt.chain_of_thought.enabled:
            current_row = preprocess_cot(current_row, cfg)
        elif cfg.prompt.react.enabled:
            current_row = preprocess_react(current_row, cfg)
        elif cfg.prompt.chaining.enabled:
            current_row = preprocess_chaining(current_row, cfg)
        
        # 4. Contextual adaptation
        if cfg.prompt.contextual.enabled:
            current_row = preprocess_contextual(current_row, cfg)
        
        # Fallback to simple prompt if no techniques enabled
        if not any([
            cfg.prompt.decision_tree.enabled,
            cfg.prompt.hierarchical.enabled,
            cfg.prompt.chain_of_thought.enabled,
            cfg.prompt.react.enabled,
            cfg.prompt.chaining.enabled,
            cfg.prompt.contextual.enabled,
            cfg.prompt.adaptive.enabled
        ]):
            return preprocess_simple(current_row, cfg)
        
        # Ensure messages are in vLLM format
        if "messages" not in current_row:
            # Default fallback
            return preprocess_simple(current_row, cfg)
        
        return {
            "messages": current_row["messages"],
            "sampling_params": current_row.get("sampling_params", cfg.sampling_params_vqa)
        }
    
    def unified_postprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        """Unified postprocessing that handles all techniques."""
        result = {
            "sample_id": row.get("sample_id"),
            "answer": row.get("generated_text"),
        }
        
        # Handle self-consistency aggregation if enabled
        if cfg.prompt.self_consistency.enabled:
            result = postprocess_self_consistency(row, cfg)
        
        # Handle hierarchical/decision tree outputs
        if cfg.prompt.hierarchical.enabled or cfg.prompt.decision_tree.enabled:
            # Extract all intermediate outputs
            metadata = row.get("metadata", {})
            if isinstance(metadata, dict):
                result.update({k: v for k, v in metadata.items() if k not in ["tree_id", "version"]})
        
        return result
    
    processor = build_llm_processor(
        processor_config,
        preprocess=unified_preprocess,
        postprocess=unified_postprocess
    )
    
    return processor

def preprocess_simple(row: Dict[str, Any], cfg: DictConfig) -> Dict[str, Any]:
    """Simple preprocessing without any dynamic techniques.
    
    Best Practice: Include error handling for missing images or invalid data.
    """
    try:
        image_content = _prepare_image_content(row.get("image"), row.get("sample_id"))
        messages = [
            {"role": "system", "content": cfg.prompt.system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": row.get("prompt", "")},
                    image_content
                ]
            }
        ]
    except Exception as e:
        # Fallback: text-only message if image loading fails
        import logging
        logging.warning(f"Failed to load image for sample {row.get('sample_id')}: {e}")
        messages = [
            {"role": "system", "content": cfg.prompt.system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": row.get("prompt", "")}
                ]
            }
        ]
    
    return {
        "messages": messages,
        "sampling_params": cfg.sampling_params_vqa
    }

def _prepare_image_content(image: Any, sample_id: str = None) -> Dict[str, Any]:
    """
    Prepare image content in vLLM-compatible OpenAI chat format.
    
    Best Practice: Returns content dict compatible with OpenAI chat API format.
    Supports PIL Images, file paths, URLs, base64 strings, and torch embeddings.
    """
    import os  # For path checking
    
    if isinstance(image, str):
        # URL (http/https)
        if image.startswith(("http://", "https://")):
            return {
                "type": "image_url",
                "image_url": {"url": image}
            }
        # Base64 encoded string
        elif image.startswith("data:image/") or (len(image) > 100 and not os.path.exists(image)):
            # Assume base64 if it's a long string and not a file path
            return {
                "type": "image_url",
                "image_url": {"url": image if image.startswith("data:") else f"data:image/jpeg;base64,{image}"}
            }
        # File path
        else:
            from PIL import Image as PILImage
            try:
                image = PILImage.open(image).convert("RGB")
            except Exception as e:
                raise ValueError(f"Failed to load image from path '{image}': {e}")
    
    # PIL Image object
    if hasattr(image, "convert") or hasattr(image, "size"):  # PIL Image
        return {"type": "image", "image": image}
    
    # Torch tensor (pre-computed embeddings) - for advanced use cases
    import torch
    if isinstance(image, torch.Tensor):
        return {"type": "image_embeds", "image_embeds": image}
    
    # Fallback: assume PIL Image
    return {"type": "image", "image": image}
```

**Structured JSON Output Support**:

Using vLLM's guided decoding with JSON schema:
```python
from pydantic import BaseModel
from typing import Optional, List

class VQAAnswer(BaseModel):
    """Structured output schema for VQA answers."""
    answer: str
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    entities: Optional[List[str]] = None
    metadata: Optional[dict] = None

def create_structured_output_processor(cfg: DictConfig):
    """Create processor with structured JSON output support."""
    from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig
    
    processor_config = vLLMEngineProcessorConfig(
        model_source=cfg.model.model_source,
        engine_kwargs={
            **cfg.model.engine_kwargs,
            "guided_decoding_backend": "xgrammar",  # or "outlines", "jsonformer", "auto"
            # Best Practice: "auto" selects backend automatically; "xgrammar" recommended for JSON
            "limit_mm_per_prompt": {"image": 1},  # Single image per prompt
            "trust_remote_code": cfg.model.get("trust_remote_code", True),  # Required for custom vision models
        },
        concurrency=cfg.model.concurrency,
        batch_size=cfg.model.batch_size,
        has_image=True,  # Best Practice: Always enable for VQA
    )
    
    # Get JSON schema from Pydantic model or config
    json_schema = VQAAnswer.model_json_schema()
    
    def preprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess with structured output requirement.
        
        Best Practice: Use guided_json in sampling_params for offline inference.
        For OpenAI API format, use response_format instead.
        """
        image_content = _prepare_image_content(row.get("image"), row.get("sample_id"))
        messages = [
            {"role": "system", "content": cfg.prompt.system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": row.get("prompt", "")},
                    image_content
                ]
            }
        ]
        
        return {
            "messages": messages,
            "sampling_params": {
                **cfg.sampling_params_vqa,
                # Best Practice: Use guided_decoding with json key for JSON schema-guided decoding
                # Format: guided_decoding={"json": json_schema} for offline inference (Ray Data)
                "guided_decoding": {"json": json_schema},  # vLLM offline inference format
                # Alternative for OpenAI API: use response_format in extra_body
                # "extra_body": {"response_format": {"type": "json_schema", "json_schema": {...}}}
            }
        }
    
    def postprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocess structured JSON output."""
        import json
        
        generated_text = row.get("generated_text", "")
        
        # Parse JSON from generated text
        try:
            # vLLM may return JSON directly or wrapped in text
            if generated_text.strip().startswith("{"):
                parsed = json.loads(generated_text)
            else:
                # Try to extract JSON from text
                import re
                json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                else:
                    # Fallback: wrap in answer field
                    parsed = {"answer": generated_text}
        except json.JSONDecodeError:
            # Fallback: wrap raw text
            parsed = {"answer": generated_text}
        
        return {
            "sample_id": row.get("sample_id"),
            **parsed  # Unpack structured fields
        }
    
    processor = build_llm_processor(
        processor_config,
        preprocess=preprocess,
        postprocess=postprocess
    )
    
    return processor
```

**Configuration for Structured Output**:
```yaml
# dagspaces/urbanvqa/conf/prompt/structured_output.yaml
prompt:
  structured_output:
    enabled: true
    schema_type: "pydantic"  # or "json_schema"
    schema_path: "dagspaces/urbanvqa/schemas/vqa_answer.py"  # Path to Pydantic model
    # OR inline JSON schema:
    json_schema:
      type: object
      properties:
        answer:
          type: string
        confidence:
          type: number
        reasoning:
          type: string
        entities:
          type: array
          items:
            type: string
      required:
        - answer

model:
  engine_kwargs:
    guided_decoding_backend: "xgrammar"  # Best Practice: "auto" (default), "xgrammar", "outlines", "jsonformer"
    # "auto" selects backend automatically based on request
    # "xgrammar" recommended for JSON schema (most performant)
    # "outlines" alternative backend for JSON
    # "jsonformer" alternative backend for JSON
    limit_mm_per_prompt: {"image": 1}  # Best Practice: Enforce single image per prompt
    trust_remote_code: true  # Best Practice: Required for custom vision models
    max_model_len: 4096  # Best Practice: Set based on model's context window
    max_num_batched_tokens: 8192  # Best Practice: Tune for throughput (default: adaptive)
```

**Hydra Configuration Groups**:
```yaml
# dagspaces/urbanvqa/conf/prompt/decision_tree.yaml
# @package _global_
prompt:
  decision_tree:
    enabled: false
    tree_path: null
    tree_format: "yaml"
    max_depth: 10
    enable_cycle_detection: true

# dagspaces/urbanvqa/conf/prompt/hierarchical.yaml
# @package _global_
prompt:
  hierarchical:
    enabled: false
    steps: []

# dagspaces/urbanvqa/conf/prompt/chain_of_thought.yaml
# @package _global_
prompt:
  chain_of_thought:
    enabled: false
    template: |
      Let's solve this step by step:
      1. First, analyze the image: {{image_analysis}}
      2. Then, reason about the question: {{reasoning}}
      3. Finally, provide the answer: {{answer}}
```

**Usage via Hydra**:
```bash
# Enable decision tree prompting
# Best Practice: Use relative paths or absolute paths, quote paths with special characters
python -m dagspaces.urbanvqa.cli \
  prompt.decision_tree.enabled=true \
  'prompt.decision_tree.tree_path=dagspaces/urbanvqa/conf/prompts/decision_trees/urban_planning_vqa.yaml'

# Enable hierarchical prompting
python -m dagspaces.urbanvqa.cli prompt.hierarchical.enabled=true

# Enable multiple techniques with priority
# Best Practice: Quote list values to prevent shell expansion
python -m dagspaces.urbanvqa.cli \
  prompt.decision_tree.enabled=true \
  prompt.chain_of_thought.enabled=true \
  'prompt.priority=["decision_tree","chain_of_thought"]'
```

### 6.6 Compatibility Notes: vLLM, Ray Data, and Hydra

**Key Compatibility Requirements**:
