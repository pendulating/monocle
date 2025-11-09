"""Dynamic prompting techniques for VQA.

This module provides various dynamic prompting techniques that can be applied
to enhance VQA reasoning capabilities:
- Chain-of-Thought (CoT)
- ReAct (Reasoning and Acting)
- Self-Consistency
- Retrieval-Augmented Prompting (RAP)
- Prompt Chaining
- Contextual Dynamic Prompting
- Adaptive Prompting
"""

from typing import Dict, Any, Optional, List
from omegaconf import DictConfig
import re


def preprocess_cot(row: Dict[str, Any], cfg: DictConfig) -> Dict[str, Any]:
    """Preprocess with Chain-of-Thought structure.
    
    Encourages the model to generate intermediate reasoning steps before
    providing the final answer.
    
    Args:
        row: Input row with prompt and image
        cfg: Configuration object
        
    Returns:
        Preprocessed row with CoT prompt structure
    """
    # Import locally to avoid circular dependencies
    from dagspaces.urbanvqa.stages.vqa import _prepare_image_content
    
    if not getattr(cfg.prompt, "chain_of_thought", None) or not getattr(cfg.prompt.chain_of_thought, "enabled", False):
        return None
    
    cot_config = cfg.prompt.chain_of_thought
    template = getattr(cot_config, "template", None)
    
    if not template:
        # Default CoT template
        template = """Let's solve this step by step:

1. First, analyze the image: {{image_analysis}}
2. Then, reason about the question: {{reasoning}}
3. Finally, provide the answer: {{answer}}"""
    
    # Build CoT prompt
    cot_prompt = template.replace("{{image_analysis}}", "Analyze the image carefully.")
    cot_prompt = cot_prompt.replace("{{reasoning}}", "Reason about the question.")
    cot_prompt = cot_prompt.replace("{{answer}}", "Provide your final answer.")
    
    # Replace user question placeholder
    user_question = row.get("prompt", "")
    cot_prompt = cot_prompt.replace("{{user_question}}", user_question)
    cot_prompt = cot_prompt.replace("{{prompt}}", user_question)
    
    # Load image
    image_content = None
    is_multimodal = getattr(getattr(cfg, "runtime", None), "multimodal_enabled", False) if hasattr(cfg, "runtime") else False
    if is_multimodal:
        try:
            image_content = _prepare_image_content(
                row.get("image") or row.get("image_path") or row.get("image_url"),
                row.get("sample_id")
            )
        except Exception:
            pass
    
    # Build vLLM-compatible messages
    if image_content:
        user_content = [
            {"type": "text", "text": cot_prompt},
            image_content
        ]
    else:
        user_content = cot_prompt
    
    system_prompt = getattr(cfg.prompt, "system", "You are a helpful assistant.")
    sampling_params = getattr(cfg, "sampling_params_vqa", {})
    
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "sampling_params": sampling_params
    }


def preprocess_react(row: Dict[str, Any], cfg: DictConfig) -> Dict[str, Any]:
    """Preprocess with ReAct (Reasoning and Acting) pattern.
    
    Combines reasoning and action steps, allowing the model to interact
    with external tools or data sources dynamically.
    
    Args:
        row: Input row with prompt and image
        cfg: Configuration object
        
    Returns:
        Preprocessed row with ReAct prompt structure
    """
    from dagspaces.urbanvqa.stages.vqa import _prepare_image_content
    
    if not getattr(cfg.prompt, "react", None) or not getattr(cfg.prompt.react, "enabled", False):
        return None
    
    react_config = cfg.prompt.react
    tools = getattr(react_config, "tools", [])
    
    # Build ReAct prompt with tool descriptions
    tool_descriptions = "\n".join([
        f"- {tool.get('name', 'tool')}: {tool.get('description', '')}"
        for tool in tools
    ])
    
    react_prompt = f"""You have access to the following tools:
{tool_descriptions}

Use these tools to answer the question. Format your reasoning as:
Thought: [your reasoning]
Action: [tool name]
Action Input: [parameters]
Observation: [tool result]
... (repeat as needed)
Final Answer: [final answer]

Question: {row.get('prompt', '')}"""
    
    # Load image
    image_content = None
    is_multimodal = getattr(getattr(cfg, "runtime", None), "multimodal_enabled", False) if hasattr(cfg, "runtime") else False
    if is_multimodal:
        try:
            image_content = _prepare_image_content(
                row.get("image") or row.get("image_path") or row.get("image_url"),
                row.get("sample_id")
            )
        except Exception:
            pass
    
    # Build vLLM-compatible messages
    if image_content:
        user_content = [
            {"type": "text", "text": react_prompt},
            image_content
        ]
    else:
        user_content = react_prompt
    
    system_prompt = getattr(cfg.prompt, "system", "You are a helpful assistant.")
    sampling_params = getattr(cfg, "sampling_params_vqa", {})
    
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "sampling_params": sampling_params
    }


def preprocess_self_consistency(row: Dict[str, Any], cfg: DictConfig) -> Dict[str, Any]:
    """Preprocess for self-consistency prompting.
    
    Generates multiple reasoning paths and selects the most consistent answer.
    Note: This requires dataset expansion to create multiple variations.
    
    Args:
        row: Input row with prompt and image
        cfg: Configuration object
        
    Returns:
        Preprocessed row with varied prompt for consistency checking
    """
    from dagspaces.urbanvqa.stages.vqa import _prepare_image_content
    
    if not getattr(cfg.prompt, "self_consistency", None) or not getattr(cfg.prompt.self_consistency, "enabled", False):
        return None
    
    sc_config = cfg.prompt.self_consistency
    variations = getattr(sc_config, "prompt_variations", [
        "Think step by step: {prompt}",
        "Analyze carefully: {prompt}",
        "Consider multiple perspectives: {prompt}"
    ])
    
    # Use first variation (dataset expansion would create multiple rows)
    base_prompt = row.get("prompt", "")
    varied_prompt = variations[0].format(prompt=base_prompt) if variations else base_prompt
    
    # Load image
    image_content = None
    is_multimodal = getattr(getattr(cfg, "runtime", None), "multimodal_enabled", False) if hasattr(cfg, "runtime") else False
    if is_multimodal:
        try:
            image_content = _prepare_image_content(
                row.get("image") or row.get("image_path") or row.get("image_url"),
                row.get("sample_id")
            )
        except Exception:
            pass
    
    # Build vLLM-compatible messages
    if image_content:
        user_content = [
            {"type": "text", "text": varied_prompt},
            image_content
        ]
    else:
        user_content = varied_prompt
    
    system_prompt = getattr(cfg.prompt, "system", "You are a helpful assistant.")
    sampling_params = getattr(cfg, "sampling_params_vqa", {})
    
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "sampling_params": sampling_params,
        "_consistency_sample_id": row.get("sample_id")  # For grouping
    }


def preprocess_rap(row: Dict[str, Any], cfg: DictConfig) -> Dict[str, Any]:
    """Preprocess with Retrieval-Augmented Prompting (RAP).
    
    Enhances prompts with relevant information retrieved from external sources.
    
    Args:
        row: Input row with prompt and image
        cfg: Configuration object
        
    Returns:
        Preprocessed row with retrieved context augmented prompt
    """
    from dagspaces.urbanvqa.stages.vqa import _prepare_image_content
    
    if not getattr(cfg.prompt, "retrieval_augmented", None) or not getattr(cfg.prompt.retrieval_augmented, "enabled", False):
        return None
    
    rap_config = cfg.prompt.retrieval_augmented
    knowledge_base = getattr(rap_config, "knowledge_base", None)
    retrieval_method = getattr(rap_config, "retrieval_method", "semantic_search")
    top_k = getattr(rap_config, "top_k", 3)
    
    # Retrieve relevant context (placeholder - would integrate with actual KB)
    context_text = ""
    if knowledge_base:
        # Placeholder: In real implementation, would call retrieval system
        # retrieved = retrieve_context(
        #     query=row["prompt"],
        #     kb=knowledge_base,
        #     method=retrieval_method,
        #     top_k=top_k
        # )
        # context_text = "\n\n".join([doc["text"] for doc in retrieved])
        pass
    
    # Build augmented prompt
    if context_text:
        augmented_prompt = f"""Use the following context to answer the question:

Context:
{context_text}

Question: {row.get('prompt', '')}"""
    else:
        augmented_prompt = row.get("prompt", "")
    
    # Load image
    image_content = None
    is_multimodal = getattr(getattr(cfg, "runtime", None), "multimodal_enabled", False) if hasattr(cfg, "runtime") else False
    if is_multimodal:
        try:
            image_content = _prepare_image_content(
                row.get("image") or row.get("image_path") or row.get("image_url"),
                row.get("sample_id")
            )
        except Exception:
            pass
    
    # Build vLLM-compatible messages
    if image_content:
        user_content = [
            {"type": "text", "text": augmented_prompt},
            image_content
        ]
    else:
        user_content = augmented_prompt
    
    system_prompt = getattr(cfg.prompt, "system", "You are a helpful assistant.")
    sampling_params = getattr(cfg, "sampling_params_vqa", {})
    
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "sampling_params": sampling_params
    }


def preprocess_chaining(row: Dict[str, Any], cfg: DictConfig) -> Dict[str, Any]:
    """Preprocess for prompt chaining.
    
    Breaks down complex tasks into a sequence of smaller, interconnected prompts.
    Note: This requires sequential processing similar to hierarchical prompts.
    
    Args:
        row: Input row with prompt and image
        cfg: Configuration object
        
    Returns:
        Preprocessed row with first step in chain (for sequential processing)
    """
    from dagspaces.urbanvqa.stages.vqa import _prepare_image_content
    
    if not getattr(cfg.prompt, "chaining", None) or not getattr(cfg.prompt.chaining, "enabled", False):
        return None
    
    chain_config = cfg.prompt.chaining
    chain_steps = getattr(chain_config, "chain", [])
    
    if not chain_steps:
        return None
    
    # Return first step (processing would be sequential like hierarchical)
    first_step = chain_steps[0]
    step_prompt = first_step.get("prompt", "")
    
    # Replace user question placeholder
    user_question = row.get("prompt", "")
    step_prompt = step_prompt.replace("{{user_question}}", user_question)
    step_prompt = step_prompt.replace("{{prompt}}", user_question)
    
    # Load image
    image_content = None
    is_multimodal = getattr(getattr(cfg, "runtime", None), "multimodal_enabled", False) if hasattr(cfg, "runtime") else False
    if is_multimodal:
        try:
            image_content = _prepare_image_content(
                row.get("image") or row.get("image_path") or row.get("image_url"),
                row.get("sample_id")
            )
        except Exception:
            pass
    
    # Build vLLM-compatible messages
    if image_content:
        user_content = [
            {"type": "text", "text": step_prompt},
            image_content
        ]
    else:
        user_content = step_prompt
    
    system_prompt = getattr(cfg.prompt, "system", "You are a helpful assistant.")
    sampling_params = getattr(cfg, "sampling_params_vqa", {})
    
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "sampling_params": sampling_params,
        "_chain_step": 0,
        "_chain_steps": chain_steps,
        "_chain_state": {"user_question": user_question}
    }


def preprocess_contextual(row: Dict[str, Any], cfg: DictConfig) -> Dict[str, Any]:
    """Preprocess with contextual dynamic prompting.
    
    Adjusts prompts in real-time based on conversation context or intermediate outputs.
    
    Args:
        row: Input row with prompt and image
        cfg: Configuration object
        
    Returns:
        Preprocessed row with contextually adapted prompt
    """
    from dagspaces.urbanvqa.stages.vqa import _prepare_image_content
    
    if not getattr(cfg.prompt, "contextual", None) or not getattr(cfg.prompt.contextual, "enabled", False):
        return None
    
    contextual_config = cfg.prompt.contextual
    adaptation_rules = getattr(contextual_config, "adaptation_rules", [])
    
    # Get context history (would use Ray actor in real implementation)
    context = []  # Placeholder
    # try:
    #     tracker = ray.get_actor("context_tracker")
    #     context = ray.get(tracker.get_context.remote(row.get("sample_id")))
    # except (ValueError, ray.exceptions.GetTimeoutError):
    #     context = []
    
    # Adapt prompt based on context
    base_prompt = row.get("prompt", "")
    adapted_prompt = base_prompt
    
    if context:
        # Check adaptation rules
        last_response = context[-1] if context else None
        if last_response and last_response.get("confidence", 1.0) < 0.5:
            # Low confidence - add detail prompt
            for rule in adaptation_rules:
                if rule.get("condition") == "low_confidence":
                    adapted_prompt = f"{base_prompt}\n\n{rule.get('detail_prompt', 'Please provide more specific details.')}"
                    break
    
    # Load image
    image_content = None
    is_multimodal = getattr(getattr(cfg, "runtime", None), "multimodal_enabled", False) if hasattr(cfg, "runtime") else False
    if is_multimodal:
        try:
            image_content = _prepare_image_content(
                row.get("image") or row.get("image_path") or row.get("image_url"),
                row.get("sample_id")
            )
        except Exception:
            pass
    
    # Build vLLM-compatible messages
    if image_content:
        user_content = [
            {"type": "text", "text": adapted_prompt},
            image_content
        ]
    else:
        user_content = adapted_prompt
    
    system_prompt = getattr(cfg.prompt, "system", "You are a helpful assistant.")
    sampling_params = getattr(cfg, "sampling_params_vqa", {})
    
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "sampling_params": sampling_params
    }


def assess_complexity(prompt: str, method: str, thresholds: Dict[str, int] = None) -> str:
    """Assess prompt complexity.
    
    Args:
        prompt: Prompt text to assess
        method: Assessment method ("token_count", "semantic_complexity", "heuristic")
        thresholds: Complexity thresholds (simple, medium, complex)
        
    Returns:
        Complexity level: "simple", "medium", or "complex"
    """
    if thresholds is None:
        thresholds = {"simple": 50, "medium": 150, "complex": 300}
    
    if method == "token_count":
        # Simple token count heuristic
        token_count = len(prompt.split())
        if token_count < thresholds.get("simple", 50):
            return "simple"
        elif token_count < thresholds.get("medium", 150):
            return "medium"
        else:
            return "complex"
    elif method == "semantic_complexity":
        # Use embedding similarity or other semantic methods
        # Placeholder - would use actual semantic analysis
        return "medium"
    elif method == "heuristic":
        # Heuristic: check for complex keywords or structures
        complex_indicators = ["analyze", "compare", "evaluate", "explain", "describe", "identify"]
        if any(indicator in prompt.lower() for indicator in complex_indicators):
            return "complex"
        elif len(prompt.split()) > 30:
            return "medium"
        else:
            return "simple"
    
    return "medium"


def preprocess_adaptive(row: Dict[str, Any], cfg: DictConfig) -> Dict[str, Any]:
    """Preprocess with adaptive prompting.
    
    Modifies prompt structures and content based on task complexity.
    
    Args:
        row: Input row with prompt and image
        cfg: Configuration object
        
    Returns:
        Preprocessed row with complexity-adapted prompt
    """
    from dagspaces.urbanvqa.stages.vqa import _prepare_image_content
    
    if not getattr(cfg.prompt, "adaptive", None) or not getattr(cfg.prompt.adaptive, "enabled", False):
        return None
    
    adaptive_config = cfg.prompt.adaptive
    complexity_assessment = getattr(adaptive_config, "complexity_assessment", {})
    prompt_variants = getattr(adaptive_config, "prompt_variants", {})
    
    # Assess complexity
    method = getattr(complexity_assessment, "method", "token_count")
    thresholds = getattr(complexity_assessment, "thresholds", {})
    complexity = assess_complexity(row.get("prompt", ""), method, thresholds)
    
    # Select prompt variant
    variant = prompt_variants.get(complexity, prompt_variants.get("medium", {}))
    variant_prompt = variant.get("prompt", row.get("prompt", ""))
    
    # Replace user question placeholder
    user_question = row.get("prompt", "")
    adapted_prompt = variant_prompt.replace("{{user_question}}", user_question)
    adapted_prompt = adapted_prompt.replace("{{prompt}}", user_question)
    
    # Load image
    image_content = None
    is_multimodal = getattr(getattr(cfg, "runtime", None), "multimodal_enabled", False) if hasattr(cfg, "runtime") else False
    if is_multimodal:
        try:
            image_content = _prepare_image_content(
                row.get("image") or row.get("image_path") or row.get("image_url"),
                row.get("sample_id")
            )
        except Exception:
            pass
    
    # Build vLLM-compatible messages
    if image_content:
        user_content = [
            {"type": "text", "text": adapted_prompt},
            image_content
        ]
    else:
        user_content = adapted_prompt
    
    system_prompt = getattr(cfg.prompt, "system", "You are a helpful assistant.")
    sampling_params = getattr(cfg, "sampling_params_vqa", {})
    
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "sampling_params": sampling_params,
        "_complexity": complexity  # For monitoring
    }
