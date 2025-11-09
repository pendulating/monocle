# Nexar Dashcam Dataset Example Usage

This document explains how to use the Nexar dashcam dataset with the UrbanVQA pipeline.

## Dataset Location

Images are stored at:
```
/share/ju/nexar_data/2023/2023-08-20/604222321527357439/frames
```

## Quick Start: Generate Parquet from Directory

The easiest way is to generate a parquet file from the directory:

```bash
# Method 1: Direct specification
# Note: Use a path accessible from GPU nodes (not /tmp)
python scripts/create_nexar_dataset.py \
    --image_dir /share/ju/nexar_data/2023/2023-08-20/604222321527357439/frames \
    --output_path /share/pierson/matt/mllmsci/data/nexar_dashcam_vqa.parquet \
    --prompt "What urban planning features are visible in this dashcam image?"

# Method 2: Using config file (uses default path from config)
python scripts/create_nexar_dataset.py \
    --config dagspaces/urbanvqa/conf/data/nexar_dashcam.yaml \
    --output_path /share/pierson/matt/mllmsci/data/nexar_dashcam_vqa.parquet
```

This creates a parquet file with columns:
- `prompt`: The question/prompt for each image
- `sample_id`: Unique identifier for each image
- `image_path`: Full path to the image file

## Step 2: Run VQA Pipeline

### Option A: Use the pre-configured pipeline (uses default path from config)

```bash
python -m dagspaces.urbanvqa.cli \
    pipeline=vqa_nexar
```

### Option B: Override parquet path

```bash
python -m dagspaces.urbanvqa.cli \
    pipeline=vqa_nexar \
    data.parquet_path=/share/pierson/matt/mllmsci/data/nexar_dashcam_vqa.parquet \
    prompt.system="You are an expert in urban planning analysis."
```

### Option C: Use config file with directory

If you've updated the config file to point to the directory, you can override the parquet path:

```bash
python -m dagspaces.urbanvqa.cli \
    pipeline=vqa_nexar \
    data.parquet_path=/share/pierson/matt/mllmsci/data/nexar_dashcam_vqa.parquet \
    data.image_path=/share/ju/nexar_data/2023/2023-08-20/604222321527357439/frames
```

## Data Configuration

The data configuration is at `dagspaces/urbanvqa/conf/data/nexar_dashcam.yaml`:

```yaml
# Path accessible from GPU nodes (not /tmp)
parquet_path: /share/pierson/matt/mllmsci/data/nexar_dashcam_vqa.parquet

columns:
  prompt: prompt
  sample_id: sample_id
  image_path: image_path

# Directory containing images
image_path: /share/ju/nexar_data/2023/2023-08-20/604222321527357439/frames

# Default prompt
default_prompt: "What urban planning features are visible in this dashcam image?"
```

## Example Prompts for Dashcam Images

Here are some example prompts you might want to use:

- "What urban planning features are visible in this dashcam image?"
- "Describe the road infrastructure visible in this image."
- "What types of buildings are visible in this dashcam image?"
- "Analyze the street layout and urban design elements."
- "What transportation infrastructure is visible?"
- "Describe the land use patterns visible in this image."

## Custom Prompts per Image

If you want different prompts for different images, you can modify the parquet file:

```python
import pandas as pd

df = pd.read_parquet('/share/pierson/matt/mllmsci/data/nexar_dashcam_vqa.parquet')

# Customize prompts based on image filename or other criteria
df.loc[df['sample_id'].str.contains('specific_pattern'), 'prompt'] = "Custom question here"

df.to_parquet('/share/pierson/matt/mllmsci/data/nexar_dashcam_vqa_custom.parquet', index=False)
```

## Advanced: Multiple Prompts per Image

If you want to ask multiple questions about the same image, you can duplicate rows:

```python
import pandas as pd

df = pd.read_parquet('/share/pierson/matt/mllmsci/data/nexar_dashcam_vqa.parquet')

# Create multiple prompts per image
prompts = [
    "What urban planning features are visible?",
    "Describe the road infrastructure.",
    "What types of buildings are visible?",
]

# Duplicate rows for each prompt
expanded_df = pd.concat([
    df.assign(prompt=p, sample_id=df['sample_id'] + f"_q{i}")
    for i, p in enumerate(prompts)
], ignore_index=True)

expanded_df.to_parquet('/share/pierson/matt/mllmsci/data/nexar_dashcam_vqa_multi.parquet', index=False)
```

## Troubleshooting

### Images not found
Make sure the paths in the parquet file are absolute and accessible from where you run the pipeline.

### Memory issues
For large datasets, enable streaming:
```bash
python -m dagspaces.urbanvqa.cli \
    pipeline=vqa_nexar \
    runtime.streaming_io=true \
    runtime.auto_streaming_threshold_gb=5.0
```

### Generating parquet on-the-fly
The script can be run as part of your workflow:
```bash
# Generate parquet and run pipeline in one command
# Note: Use a path accessible from GPU nodes (not /tmp)
python scripts/create_nexar_dataset.py \
    --config dagspaces/urbanvqa/conf/data/nexar_dashcam.yaml \
    --output_path /share/pierson/matt/mllmsci/data/nexar_dashcam_vqa.parquet && \
python -m dagspaces.urbanvqa.cli \
    pipeline=vqa_nexar \
    data.parquet_path=/share/pierson/matt/mllmsci/data/nexar_dashcam_vqa.parquet
```
