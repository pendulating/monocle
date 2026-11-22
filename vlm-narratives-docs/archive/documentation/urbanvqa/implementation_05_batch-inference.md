
## Phase 5: Batch Inference Support

### 5.1 Batch Inference Support

**Requirement**: Support processing multiple independent prompt+image pairs efficiently via Ray Data batch processing.

**Implementation**:

1. **Batch Processing**:
   - Each row contains an independent prompt+image pair
   - Ray Data processes multiple pairs in batches for efficient GPU utilization
   - Each sample is processed independently

2. **Prompt Grouping Optimization** (for efficiency):
   - If multiple images share the same prompt, group them together within a batch
   - Avoids retokenizing the same prompt multiple times
   - Improves throughput when the same question is asked about multiple images
   - Implementation: Pre-group by prompt before batching, then process grouped items together

   ```python
   def _group_by_prompt_optimization(df: pd.DataFrame) -> pd.DataFrame:
       """
       Group rows by prompt for efficient batch processing.
       
       This optimization reorders rows so that rows with the same prompt
       are processed together, allowing prompt tokenization to be reused.
       
       Best Practice: Apply this before creating Ray Dataset to ensure
       same prompts are batched together within Ray Data's batch processing.
       """
       # Add a grouping key
       df['_prompt_group'] = df.groupby('prompt').ngroup()
       # Sort by prompt group to ensure same prompts are batched together
       df = df.sort_values('_prompt_group').reset_index(drop=True)
       return df
   
   # Apply before creating Ray Dataset
   df_grouped = _group_by_prompt_optimization(df)
   ds = ray.data.from_pandas(df_grouped)
   ```

3. **Configuration**:
   ```yaml
   runtime:
     batch_inference: true  # Enable batch processing
     batch_size: 16  # Number of prompt+image pairs per batch
     group_by_prompt: true  # Optional: Group same prompts together for efficiency
   ```

**Note**: 
- When `group_by_prompt: true`, rows with the same prompt are grouped together within batches, enabling prompt tokenization reuse.
- When `group_by_prompt: false` (default), rows are processed in order without grouping, suitable for fully independent prompt+image pairs.
- The batch size controls how many prompt+image pairs are processed together for GPU efficiency.
- Each pair remains independent regardless of grouping strategy.

### 5.2 Output Format

**Expected Output Schema**:
```python
{
    "sample_id": str,  # Unique identifier
    "prompt": str,  # Original prompt
    "image_path": str,  # Image source (path/URL/base64)
    "answer": str,  # Model's answer
    "model_response": str,  # Full model response
    "metadata": dict,  # Additional metadata (tokens, timing, etc.)
}
```