"""Full-pipeline DP worker — preprocess + infer + postprocess in-worker.

Launch protocol: the parent process runs this file directly, as

    python <path-to-this-file> <task_pickle> <result_pickle>

with a per-rank child environment. The child needs no package
importability, because this module imports only stdlib names at load time.
Heavy imports (vllm, PIL, pandas, cloudpickle) happen inside ``main()``.

Processes rows in fixed-size chunks. For each chunk:
  1. preprocess_fn is called per row (cheap — builds the message dict)
  2. images are decoded + resized in parallel CPU threads
  3. llm.chat() is called once on the chunk
  4. results are postprocessed and accumulated
  5. chunk-local objects are dropped before the next iteration

This bounds engine-core memory (no big pre-rendered prompt buffer) and gives
us streaming progress + ETA every ~``log_every`` rows. Critical for large
multimodal jobs (e.g. ~120k pairs per worker) where the previous unbounded
``llm.chat(big_list)`` path spent hours in single-threaded HF processor work
and OOM-killed the engine core.

Each worker receives raw row dicts and cloudpickled preprocess/postprocess
callables. Preprocessing happens locally in the worker, co-located with
the GPU.
"""
import os, pickle, sys, time, traceback

def main():
    task_path = sys.argv[1]
    result_path = sys.argv[2]

    with open(task_path, "rb") as f:
        task = pickle.load(f)

    rank        = task["rank"]
    dp_size     = task["dp_size"]
    engine_kwargs = task["engine_kwargs"]
    rows        = task["rows"]
    stage_name  = task["stage_name"]
    pcie_env    = task["pcie_env"]
    runtime_env = task["runtime_env"]
    model_source = task["model_source"]
    thinking_enabled = task["thinking_enabled"]
    chunk_size  = int(task.get("chunk_size", 64) or 64)
    log_every   = int(task.get("log_every", 1000) or 1000)
    image_max_pixels = task.get("image_max_pixels")  # Optional[int]
    image_load_workers = int(task.get("image_load_workers", 16) or 16)
    flush_every = int(task.get("flush_every", 1000) or 1000)
    streaming_output_dir = task.get("streaming_output_dir")  # Optional[str]
    row_offset  = int(task.get("row_offset", 0) or 0)
    image_pack_dir = task.get("image_pack_dir")  # Optional[str]

    # Deserialize preprocess/postprocess via cloudpickle
    import cloudpickle
    preprocess_fn = cloudpickle.loads(task["preprocess_bytes"])
    postprocess_fn = cloudpickle.loads(task["postprocess_bytes"])

    # Apply env vars BEFORE any CUDA/torch import
    for k, v in {**pcie_env, **runtime_env}.items():
        os.environ.setdefault(k, v)

    for var in ("VLLM_DP_RANK", "VLLM_DP_RANK_LOCAL", "VLLM_DP_SIZE",
                "VLLM_DP_MASTER_IP", "VLLM_DP_MASTER_PORT"):
        os.environ.pop(var, None)

    print(f"[{stage_name}] DP rank {rank}/{dp_size}: starting "
          f"(pid={os.getpid()}, CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}, "
          f"rows={len(rows)}, chunk_size={chunk_size}, "
          f"flush_every={flush_every}, "
          f"image_max_pixels={image_max_pixels}, "
          f"image_load_workers={image_load_workers}, "
          f"streaming_output_dir={streaming_output_dir})", flush=True)

    if streaming_output_dir:
        os.makedirs(streaming_output_dir, exist_ok=True)

    llm = None
    try:
        t0 = time.time()
        from vllm import LLM, SamplingParams
        from PIL import Image
        from concurrent.futures import ThreadPoolExecutor
        from io import BytesIO
        import re
        import math
        from dagspaces.common import image_pack as _image_pack
        if image_pack_dir:
            _pack = _image_pack.activate(image_pack_dir)
            print(f"[{stage_name}] DP rank {rank}/{dp_size}: image pack ON: "
                  f"{image_pack_dir} ({len(_pack)} images in the index)", flush=True)

        print(f"[{stage_name}] DP rank {rank}/{dp_size}: vLLM imported in "
              f"{time.time() - t0:.1f}s, creating LLM engine...", flush=True)

        t1 = time.time()
        # Allow loading local images via file:// URLs in llm.chat() (still
        # needed when callers pass image_url blocks; we usually rewrite
        # them to image_pil blocks below).
        engine_kwargs.setdefault("allowed_local_media_path", "/")
        llm = LLM(**engine_kwargs)
        print(f"[{stage_name}] DP rank {rank}/{dp_size}: LLM created in "
              f"{time.time() - t1:.1f}s, processing {len(rows)} rows in "
              f"chunks of {chunk_size}...", flush=True)

        # ─── Helpers ────────────────────────────────────────────────────
        def _strip_thinking(text):
            text = re.sub(r"<think>[\s\S]*?</think>", "", text)
            text = re.sub(r"<think>[\s\S]*$", "", text)
            return text.strip()

        def _extract_reasoning(raw_text):
            reasoning, content = "", raw_text
            if not thinking_enabled:
                return reasoning, content
            if "<think>" in raw_text:
                m = re.search(r"<think>([\s\S]*?)</think>([\s\S]*)", raw_text)
                if m:
                    reasoning = m.group(1).strip()
                    content = m.group(2).strip()
                else:
                    content = _strip_thinking(raw_text)
            elif "</think>" in raw_text:
                parts = raw_text.split("</think>", 1)
                reasoning = parts[0].strip()
                content = parts[1].strip() if len(parts) > 1 else ""
            return reasoning, content

        def _resize_to_max_pixels(img, max_pixels):
            """Downscale a PIL image so width*height <= max_pixels."""
            if max_pixels is None or max_pixels <= 0:
                return img
            w, h = img.size
            cur = w * h
            if cur <= max_pixels:
                return img
            scale = math.sqrt(max_pixels / cur)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            return img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        def _load_one_image(path):
            try:
                # An active image pack gives the same bytes with 1 NFS read
                # and no directory work. See dagspaces/common/image_pack.py.
                data = _image_pack.read_image_bytes(path)
                with Image.open(BytesIO(data) if data is not None else path) as im:
                    im.load()
                    img = im.convert("RGB")
                if image_max_pixels:
                    img = _resize_to_max_pixels(img, image_max_pixels)
                return img
            except Exception as e:
                return e  # caller treats Exception as a load failure

        # Collect (msg_block, path) tuples that need lazy loading. The
        # block is the dict we'll mutate in-place to attach the loaded
        # PIL Image. Supports image_url (file://), image (with str path
        # or PIL), and image_pil blocks.
        def _collect_image_blocks(messages):
            tasks = []  # list of (block_dict, str_path)
            for msg in messages:
                content = msg.get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    if btype == "image_url":
                        url = (block.get("image_url") or {}).get("url", "")
                        path = url.removeprefix("file://") if isinstance(url, str) else ""
                        if path:
                            tasks.append((block, path))
                    elif btype == "image":
                        val = block.get("image")
                        if isinstance(val, str):
                            path = val.removeprefix("file://")
                            tasks.append((block, path))
                        # if it's already a PIL.Image, leave it; the
                        # _convert_image_blocks pass below will rename
                        # type → image_pil for vLLM.
                    elif btype == "image_pil":
                        pass  # already loaded
            return tasks

        def _convert_image_blocks(messages):
            """Normalize image blocks to vLLM's image_pil format."""
            for msg in messages:
                content = msg.get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    if btype == "image" and "image" in block:
                        val = block["image"]
                        if hasattr(val, "size"):  # PIL.Image
                            block["type"] = "image_pil"
                            block["image_pil"] = block.pop("image")
            return messages

        thread_pool = ThreadPoolExecutor(max_workers=image_load_workers)

        def _hydrate_chunk_images(chunk_messages):
            """Decode + resize all image paths for a chunk in parallel."""
            all_tasks = []  # list of (block, path)
            for msgs in chunk_messages:
                all_tasks.extend(_collect_image_blocks(msgs))
            if not all_tasks:
                return 0
            paths = [p for _, p in all_tasks]
            blocks = [b for b, _ in all_tasks]
            pack = _image_pack.active()
            if pack is not None:
                # Bulk reads in offset order; the cache holds 1 chunk.
                pack.drop_cache()
                pack.prefetch(paths, workers=image_load_workers)
            results = list(thread_pool.map(_load_one_image, paths))
            n_failed = 0
            for block, img in zip(blocks, results):
                if isinstance(img, Exception):
                    block["__image_load_error__"] = str(img)
                    n_failed += 1
                    continue
                # Replace whatever was here with image_pil
                block.pop("image_url", None)
                block.pop("image", None)
                block["type"] = "image_pil"
                block["image_pil"] = img
            return n_failed

        # ─── First pass: build SamplingParams from the first valid row ──
        # One SamplingParams object serves every row of this shard. The
        # chunk loop below compares each preprocessed row against the
        # first valid row and stops with an error when the params differ.
        sampling_params = None
        first_pp_row = None
        _first_sp = None
        first_valid_idx = 0
        peek_idx = 0
        while peek_idx < len(rows) and first_pp_row is None:
            try:
                first_pp_row = preprocess_fn(rows[peek_idx])
            except Exception:
                first_pp_row = None
            if first_pp_row is not None:
                first_valid_idx = peek_idx
                _first_sp = first_pp_row.get("sampling_params") or {}
            peek_idx += 1
        if first_pp_row is not None:
            sp = dict(_first_sp)
            guided = sp.pop("guided_decoding", None) or sp.pop("structured_output", None)
            for k in ("early_stopping", "length_penalty", "response_format", "detokenize"):
                sp.pop(k, None)
            if guided and isinstance(guided, dict):
                try:
                    from vllm.sampling_params import StructuredOutputsParams
                    sp["structured_outputs"] = StructuredOutputsParams(**guided)
                except ImportError:
                    try:
                        from vllm.sampling_params import GuidedDecodingParams
                        sp["guided_decoding"] = GuidedDecodingParams(**guided)
                    except ImportError:
                        pass
            sampling_params = SamplingParams(**sp)
        else:
            sampling_params = SamplingParams()

        chat_kwargs = {}
        if thinking_enabled:
            chat_kwargs["chat_template_kwargs"] = {"enable_thinking": True}

        # ─── Chunked main loop ──────────────────────────────────────────
        total = len(rows)
        results = [None] * total  # filled in original order
        processed = 0
        last_log_at = 0
        n_failed_preprocess = 0
        n_failed_image_load = 0

        # Streaming parquet flushing.  We track the next row index that
        # has not yet been written to a shard.  Each chunk's results are
        # always contiguous from chunk_start..chunk_end, so a flush boundary
        # is reached whenever (processed - flushed_upto) >= flush_every.
        flushed_upto = 0
        flush_part_idx = 0
        try:
            import pandas as _pd
        except Exception:
            _pd = None  # parquet shards become a no-op if pandas missing

        def _flush_shard(end_idx):
            """Write results[flushed_upto:end_idx] to a parquet shard."""
            nonlocal flushed_upto, flush_part_idx
            if not streaming_output_dir or _pd is None:
                flushed_upto = end_idx
                return
            slab = results[flushed_upto:end_idx]
            slab = [r for r in slab if r is not None]
            if not slab:
                flushed_upto = end_idx
                return
            shard_name = (
                f"rank{rank:02d}_part{flush_part_idx:04d}_"
                f"rows{flushed_upto + row_offset:08d}-"
                f"{end_idx + row_offset:08d}.parquet"
            )
            shard_path = os.path.join(streaming_output_dir, shard_name)
            try:
                df_shard = _pd.DataFrame(slab)
                df_shard.to_parquet(shard_path, index=False, compression="snappy")
                print(
                    f"[{stage_name}] DP rank {rank}/{dp_size}: "
                    f"flushed {len(slab)} rows → {shard_path}",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[{stage_name}] DP rank {rank}/{dp_size}: "
                    f"streaming flush FAILED ({type(e).__name__}: {e})",
                    flush=True,
                )
            flush_part_idx += 1
            flushed_upto = end_idx

        for chunk_start in range(0, total, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total)
            chunk_rows = rows[chunk_start:chunk_end]

            t_chunk = time.time()
            # Pre-chunk heartbeat: appears immediately so the user can see
            # the worker is alive even on the cold-start chunk (where vLLM's
            # first llm.chat() does extra one-time work).
            print(
                f"[{stage_name}] DP rank {rank}/{dp_size}: "
                f"starting chunk {chunk_start // chunk_size + 1}"
                f"/{(total + chunk_size - 1) // chunk_size} "
                f"(rows {chunk_start}–{chunk_end - 1})",
                flush=True,
            )

            # Preprocess (build messages for each row)
            chunk_pp_rows = []         # parallel to chunk_rows
            chunk_messages = []        # parallel to a subset of valid rows
            valid_indices_local = []   # indices into chunk_rows

            for ci, row in enumerate(chunk_rows):
                try:
                    pp = preprocess_fn(row)
                except Exception as e:
                    if type(e).__name__ == "ImageLoadError":
                        row["__image_load_error__"] = str(e)
                    row["__preprocess_error__"] = str(e)
                    row["generated_text"] = ""
                    row["generated_reasoning"] = ""
                    chunk_pp_rows.append(row)
                    n_failed_preprocess += 1
                    continue
                chunk_pp_rows.append(pp)
                msgs = pp.get("messages")
                if not msgs:
                    pp["generated_text"] = ""
                    pp["generated_reasoning"] = ""
                    n_failed_preprocess += 1
                    continue

                # Fail fast: this worker sends one SamplingParams object to
                # llm.chat() for the whole shard. Contract: when
                # data_parallel_size > 1, post-preprocess sampling_params
                # must be uniform across all rows of a shard. A key that
                # toggles between absent and None also counts as a
                # difference. Stop on the first other-params row, instead
                # of a silent wrong sample.
                if _first_sp is not None:
                    row_sp = pp.get("sampling_params") or {}
                    if row_sp != _first_sp:
                        diff_keys = sorted(
                            k for k in set(row_sp) | set(_first_sp)
                            if row_sp.get(k) != _first_sp.get(k)
                        )
                        raise ValueError(
                            f"[{stage_name}] DP rank {rank}/{dp_size}: "
                            f"sampling_params differ on dataset row "
                            f"{row_offset + chunk_start + ci} "
                            f"(shard offset {chunk_start + ci}); keys "
                            f"{diff_keys} differ from the first valid row "
                            f"(dataset row {row_offset + first_valid_idx}). "
                            f"The DP full path sends one SamplingParams "
                            f"object to llm.chat() for the whole shard. "
                            f"Make the per-row params uniform, or run this "
                            f"stage with data_parallel_size=1."
                        )

                _convert_image_blocks(msgs)
                chunk_messages.append(msgs)
                valid_indices_local.append(ci)

            # Parallel image decode + resize for the chunk
            t_img = time.time()
            n_img_fail = _hydrate_chunk_images(chunk_messages)
            n_failed_image_load += n_img_fail
            img_secs = time.time() - t_img

            # Drop conversations whose image loads failed (any block in
            # the conversation got tagged with __image_load_error__)
            ok_messages = []
            ok_local_indices = []
            for ci, msgs in zip(valid_indices_local, chunk_messages):
                bad = False
                for m in msgs:
                    cnt = m.get("content")
                    if not isinstance(cnt, list):
                        continue
                    for blk in cnt:
                        if isinstance(blk, dict) and "__image_load_error__" in blk:
                            bad = True
                            break
                    if bad:
                        break
                if bad:
                    chunk_pp_rows[ci]["__image_load_error__"] = "image decode failed"
                    chunk_pp_rows[ci]["generated_text"] = ""
                    chunk_pp_rows[ci]["generated_reasoning"] = ""
                else:
                    ok_messages.append(msgs)
                    ok_local_indices.append(ci)

            # Generate. tqdm is disabled because vLLM's per-conversation
            # progress bar floods stderr at multi-line-per-row cadence on
            # SLURM (no carriage-return collapsing).  Our per-chunk and
            # per-flush log lines + the pre-chunk "starting chunk N" line
            # provide enough visibility.
            t_gen = time.time()
            if ok_messages:
                outputs = llm.chat(
                    ok_messages,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                    **chat_kwargs,
                )
            else:
                outputs = []
            gen_secs = time.time() - t_gen

            # Postprocess this chunk and place into the results array
            ok_local_set = set(ok_local_indices)
            out_iter = iter(outputs)
            for ci, pp_row in enumerate(chunk_pp_rows):
                if ci in ok_local_set:
                    out = next(out_iter)
                    raw_text = out.outputs[0].text if out.outputs else ""
                    reasoning, content = _extract_reasoning(raw_text)
                    pp_row["generated_text"] = content
                    pp_row["generated_reasoning"] = reasoning
                    pt = len(out.prompt_token_ids) if out.prompt_token_ids else 0
                    ct = (len(out.outputs[0].token_ids)
                          if out.outputs and out.outputs[0].token_ids else 0)
                    pp_row["usage"] = {
                        "prompt_tokens": pt,
                        "completion_tokens": ct,
                        "total_tokens": pt + ct,
                    }
                try:
                    results[chunk_start + ci] = postprocess_fn(pp_row)
                except Exception as e:
                    pp_row["__postprocess_error__"] = str(e)
                    results[chunk_start + ci] = pp_row

            # Free chunk-local objects so RSS does not grow
            del chunk_rows, chunk_pp_rows, chunk_messages, ok_messages, outputs
            processed = chunk_end

            # Streaming progress: log every chunk so users always see
            # forward motion. log_every is only used as a hint for the
            # default chunk_size — actual reporting cadence is per-chunk.
            chunk_secs = time.time() - t_chunk
            elapsed = time.time() - t1
            rate = processed / max(elapsed, 1e-6)
            eta = (total - processed) / max(rate, 1e-6)
            print(
                f"[{stage_name}] DP rank {rank}/{dp_size}: "
                f"{processed}/{total} ({100.0 * processed / total:.1f}%) "
                f"| chunk {chunk_secs:.1f}s "
                f"(img {img_secs:.1f}s, gen {gen_secs:.1f}s) "
                f"| {rate:.2f} rows/s | elapsed {elapsed:.0f}s "
                f"| ETA {eta:.0f}s",
                flush=True,
            )
            last_log_at = processed

            # Streaming parquet flush whenever we have at least
            # flush_every newly-completed rows since the last shard.
            if processed - flushed_upto >= flush_every:
                _flush_shard(processed)

        thread_pool.shutdown(wait=False)

        # Sanity: every slot should be filled
        for i, r in enumerate(results):
            if r is None:
                results[i] = rows[i]

        # Final flush of any rows that did not cross the flush_every boundary
        if flushed_upto < total:
            _flush_shard(total)

        print(f"[{stage_name}] DP rank {rank}/{dp_size}: done, "
              f"{len(results)} results, "
              f"preprocess_failed={n_failed_preprocess}, "
              f"image_load_failed={n_failed_image_load}, "
              f"{_image_pack.stats_line()}, "
              f"total elapsed {time.time() - t0:.1f}s", flush=True)

        with open(result_path, "wb") as f:
            pickle.dump({"rank": rank, "results": results, "error": None}, f)
        return

    except Exception:
        tb = traceback.format_exc()
        print(f"[{stage_name}] DP rank {rank}/{dp_size}: FAILED\n{tb}",
              flush=True, file=sys.stderr)
        with open(result_path, "wb") as f:
            pickle.dump({"rank": rank, "results": None, "error": tb}, f)
        sys.exit(1)
    finally:
        try:
            if llm is not None:
                engine = getattr(llm, "llm_engine", None)
                core = getattr(engine, "engine_core", None) if engine else None
                if core is not None and hasattr(core, "shutdown"):
                    core.shutdown()
                del llm
        except Exception:
            pass

if __name__ == "__main__":
    main()
