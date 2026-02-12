import argparse
import pandas as pd
import torch
from transformers import pipeline
from tqdm.auto import tqdm
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def classify_batch(batch, model_name, labels, hypothesis_template, device, worker_id=0, batch_size=16):
    """Worker function for parallel classification."""
    # Re-initialize classifier in each process
    classifier = pipeline("zero-shot-classification", model=model_name, device=device)
    
    results = []
    # Process in smaller batches within the worker to show progress
    for i in tqdm(range(0, len(batch), batch_size), desc=f"Worker {worker_id}", position=worker_id + 1, leave=False):
        sub_batch = batch[i:i+batch_size]
        batch_results = classifier(
            sub_batch, 
            labels, 
            hypothesis_template=hypothesis_template
        )
        if isinstance(batch_results, dict):
            batch_results = [batch_results]
        results.extend(batch_results)
    return results

def classify_flood_response(df, classifier, labels, batch_size=16, model_name=None, num_workers=1):
    """
    Classifies the 'model_response' column in a DataFrame using a zero-shot classifier.
    Supports parallelization across multiple GPUs or CPUs.
    """
    responses = df['model_response'].astype(str).tolist()
    
    if num_workers <= 1:
        results = []
        for i in tqdm(range(0, len(responses), batch_size), desc="Classifying responses (serial)"):
            batch = responses[i:i+batch_size]
            batch_results = classifier(
                batch, 
                labels, 
                hypothesis_template="The answer to whether there is a flood is {}."
            )
            if isinstance(batch_results, dict):
                batch_results = [batch_results]
            results.extend(batch_results)
    else:
        print(f"Parallelizing across {num_workers} workers...")
        # Split responses into chunks for each worker
        chunks = np.array_split(responses, num_workers)
        
        # Determine devices for each worker
        num_gpus = torch.cuda.device_count()
        worker_args = []
        for i, chunk in enumerate(chunks):
            if len(chunk) == 0: continue
            device = i % num_gpus if num_gpus > 0 else -1
            worker_args.append((
                chunk.tolist(), 
                model_name, 
                labels, 
                "The answer to whether there is a flood is {}.",
                device,
                i,
                batch_size
            ))
            
        results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(classify_batch, *args) for args in worker_args]
            for future in tqdm(futures, desc="Overall progress (chunks)"):
                results.extend(future.result())
    
    # Extract the top label and score
    df['nli_label'] = [r['labels'][0] for r in results]
    df['nli_score'] = [r['scores'][0] for r in results]
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Classify VQA responses using NLI for flood detection.")
    parser.add_argument("--input", type=str, required=True, help="Path to input parquet file.")
    parser.add_argument("--output", type=str, required=True, help="Path to output parquet file.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for classification.")
    parser.add_argument("--model", type=str, default="facebook/bart-large-mnli", help="HuggingFace model to use.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of rows to process (for testing).")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel workers.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        return

    print(f"Loading data from {args.input}...")
    df = pd.read_parquet(args.input)
    
    if args.limit:
        print(f"Limiting to first {args.limit} rows.")
        df = df.head(args.limit).copy()
    
    if 'model_response' not in df.columns:
        print(f"Error: 'model_response' column not found in {args.input}")
        print(f"Available columns: {list(df.columns)}")
        return

    print(f"Initializing classifier with model {args.model}...")
    device = 0 if torch.cuda.is_available() else -1
    
    # Only initialize main classifier if not parallelizing
    classifier = None
    if args.num_workers <= 1:
        classifier = pipeline("zero-shot-classification", model=args.model, device=device)
    
    candidate_labels = ["no, not passable", "yes, passable", "uncertain", "not applicable"]
    
    print("Starting classification...")
    df = classify_flood_response(
        df, 
        classifier, 
        candidate_labels, 
        batch_size=args.batch_size,
        model_name=args.model,
        num_workers=args.num_workers
    )
    
    print(f"Saving results to {args.output}...")
    # Ensure the output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    df.to_parquet(args.output)
    print(f"Successfully processed {len(df)} rows and saved to {args.output}")

if __name__ == "__main__":
    # Use spawn for CUDA compatibility in multiprocessing
    if torch.cuda.is_available():
        multiprocessing.set_start_method('spawn', force=True)
    main()


