"""Decision tree prompt support for hierarchical and adaptive VQA reasoning.

This module provides decision tree functionality for VQA tasks where the next
question depends on previous answers, enabling adaptive and context-sensitive
interactions.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import json
import yaml
import re
import os


@dataclass
class TreeNode:
    """Represents a node in a decision tree."""
    node_id: str
    prompt: str
    node_type: str  # "question", "decision", "leaf", "action", "convergence"
    output_key: Optional[str] = None
    branches: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DecisionTree:
    """Manages decision tree structure and navigation."""
    
    def __init__(self, tree_config: Dict[str, Any]):
        """Initialize decision tree from configuration dict.
        
        Args:
            tree_config: Dictionary containing tree_id, root_node, and nodes
        """
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
        """Load decision tree from JSON file.
        
        Args:
            json_path: Path to JSON file containing tree definition
            
        Returns:
            DecisionTree instance
        """
        with open(json_path, "r") as f:
            config = json.load(f)
        return cls(config)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DecisionTree":
        """Load decision tree from YAML file.
        
        Args:
            yaml_path: Path to YAML file containing tree definition
            
        Returns:
            DecisionTree instance
        """
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(config)
    
    def get_next_node(self, current_node_id: str, model_response: str, 
                     context: Dict[str, Any] = None) -> Optional[str]:
        """Determine next node based on current node and model response.
        
        Args:
            current_node_id: ID of current node
            model_response: Model's response text
            context: Optional context dictionary for condition evaluation
            
        Returns:
            Next node ID or None if no valid branch found
        """
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
        """Evaluate a branch condition.
        
        Args:
            condition: Condition dictionary with 'type' and condition-specific fields
            response: Model response text to evaluate
            context: Optional context for condition evaluation
            
        Returns:
            True if condition matches, False otherwise
        """
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
            return bool(re.search(pattern, response, re.IGNORECASE))
        
        elif cond_type == "always":
            return True
        
        elif cond_type == "default":
            return True  # Default branch matches if no other conditions match
        
        return False
    
    def get_convergence_inputs(self, node_id: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Get inputs from all predecessor nodes that feed into a convergence node.
        
        Args:
            node_id: ID of convergence node
            results: Dictionary of current results (output_key -> value)
            
        Returns:
            Dictionary of inputs from predecessor nodes
        """
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
        """Aggregate multiple inputs according to strategy.
        
        Args:
            inputs: Dictionary of inputs to aggregate
            strategy: Aggregation strategy ("concatenate", "summarize", "list", "json")
            
        Returns:
            Aggregated string representation
        """
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
            return json.dumps(inputs, indent=2)
        else:
            # Default to concatenate
            return "\n\n".join([f"{k}: {v}" for k, v in inputs.items()])
    
    def _find_predecessors(self, node_id: str) -> List[str]:
        """Find all predecessor nodes that lead to a given node.
        
        Args:
            node_id: Target node ID
            
        Returns:
            List of predecessor node IDs
        """
        predecessors = []
        for pred_node_id, pred_node in self.nodes.items():
            if pred_node.branches:
                for branch in pred_node.branches:
                    if branch.get("target_node") == node_id:
                        predecessors.append(pred_node_id)
        return predecessors
