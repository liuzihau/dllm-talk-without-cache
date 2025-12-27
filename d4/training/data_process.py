from typing import Any, Dict, List, Optional, Union
import torch
from datasets import load_dataset, load_from_disk


import os
import re
from pathlib import Path
from datasets import load_dataset, load_from_disk

def build_dataset_rank(
    tokenizer,
    datapath: str,
    max_len: int,
    *,
    split: str = "chat",
    cache_root: str = "./hf_datasets_cache",
    num_proc: int = 8,
):
    """
    datapath:
      - local path to a dataset saved with save_to_disk(), OR
      - HF Hub dataset name (optionally with config, e.g. "org/name" or "org/name:config")

    Behavior:
      - If local path exists -> load_from_disk(datapath)
      - Else -> download via load_dataset(...) and save_to_disk(...) under cache_root,
                then load_from_disk(...) (so next run is offline / faster)
    """

    def _is_local_saved_dataset(p: str) -> bool:
        # load_from_disk expects a directory created by save_to_disk()
        # Check common marker files/dirs.
        path = Path(p)
        if not path.exists() or not path.is_dir():
            return False
        return (path / "dataset_info.json").exists() or (path / "state.json").exists() or (path / "data").exists()

    def _parse_hf_id(s: str):
        # Allow "repo_id" or "repo_id:config"
        if ":" in s:
            repo_id, config = s.split(":", 1)
            repo_id, config = repo_id.strip(), config.strip()
            return repo_id, (config if config else None)
        return s.strip(), None

    def _safe_dirname(s: str) -> str:
        # filesystem-safe stable name
        s = s.strip()
        s = re.sub(r"[^\w\-.]+", "_", s)
        return s

    # 1) Resolve dataset source -> local on-disk path
    if _is_local_saved_dataset(datapath):
        local_path = Path(datapath)
        ds = load_from_disk(str(local_path))
    else:
        # Not a local saved dataset; treat as HF dataset id and cache it to disk.
        repo_id, config = _parse_hf_id(datapath)

        cache_root = Path(cache_root)
        cache_root.mkdir(parents=True, exist_ok=True)

        cache_key = repo_id if config is None else f"{repo_id}:{config}"
        local_path = cache_root / _safe_dirname(cache_key) / split

        if _is_local_saved_dataset(str(local_path)):
            ds = load_from_disk(str(local_path))
        else:
            # Download from HF and persist
            if config is None:
                ds_hf = load_dataset(repo_id, split=split)
            else:
                ds_hf = load_dataset(repo_id, config, split=split)

            local_path.mkdir(parents=True, exist_ok=True)
            ds_hf.save_to_disk(str(local_path))
            ds = load_from_disk(str(local_path))

    # 2) Normal pipeline (same spirit as EAGLE3)
    ds = ds.shuffle(seed=42)
    ds1 = ds
    original_columns1 = ds1.column_names

    def preprocess_function(examples):
        new_examples = {
            "attention_mask": [],
            "target": [],
            "input_ids": []
        }
        for i in range(len(examples)):
            messages = [
                {"role": "system",
                 "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
            ]
            convroles = ["user", "assistant"]
            roles = {"human", "user", "gpt", "assistant"}
            source = examples['input'][i]
            response = examples['output'][i]
            if not source or source[0]["role"] not in roles:
                continue
            # if len(source) > 1:
            #     print(len(source))
            #     print(source)
            for msg in source:
                messages.append(
                    {"role": msg["role"], "content": msg["content"]}
                )
            messages.append(
                    {"role": "assistant", "content": response}
                )
            conversation = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            ).removesuffix("<|start_header_id|>assistant<|end_header_id|>\n\n")

            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id = tokenizer.unk_token_id

            full_ids = tokenizer(
                conversation,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids[0]

            # filtering out the samples which is longer than max_len
            if len(full_ids) > max_len:
                continue
            
            
            sep = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            turns = conversation.split(sep)
            if len(turns) < 2:
                continue

            prompt = ""
            for turn in turns[:-1]:
                prompt += turn + sep
            input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
            target = full_ids[len(input_ids):]

            attention_mask = torch.ones_like(input_ids)

            # new_examples["conversation"].append(conversation)
            new_examples["input_ids"].append(input_ids[None, :])
            new_examples["target"].append(target[None, :])
            new_examples["attention_mask"].append(attention_mask[None, :])

        return new_examples

    ds1 = ds1.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns1,
        load_from_cache_file=False,
    )

    ds1.set_format(type="torch")
    return ds1


class DataCollatorWithPadding:

    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        # padding_tensor = torch.zeros(B, N - n, S,dtype=intensors.dtype)
        padding_tensor = torch.zeros(B, N - n, S, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]], length: int = 4, mask_token_id: int = 126336, pad_token_id: int = 126081) -> Dict[str, Any]:
        # ---- helpers: accept [T] or [1,T] and always use [T]
        def _to_1d(x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 2 and x.size(0) == 1:
                return x.squeeze(0)
            return x

        B = len(features)
        device = features[0]["input_ids"].device

        input_ids_list = [_to_1d(f["input_ids"]) for f in features]          # [Li]
        attn_mask_list = [_to_1d(f["attention_mask"]) for f in features]     # [Li]
        target_list    = [_to_1d(f["target"]) for f in features]             # [Ti]

        # ---- sample start indices uniformly for each example
        # valid starts: 0..Ti-length (inclusive) => count = Ti-length+1
        max_starts = torch.tensor(
            [max(t.size(0) - length + 1, 1) for t in target_list],
            device=device
        )  # [B], clamp to >=1 to avoid errors if Ti < length

        # uniform integer in [0, max_starts[i)-1]
        starts = (torch.rand(B, device=device) * max_starts).long()  # [B]

        # ---- compute final sequence lengths and max_length
        seq_lens = []
        for i in range(B):
            prefix_len = int(starts[i].item())
            seq_lens.append(input_ids_list[i].size(0) + prefix_len + length)
        max_length = max(seq_lens)

        # ---- allocate batch tensors (more efficient than per-item pad+cat)
        dtype_ids = input_ids_list[0].dtype  # usually torch.long
        batch_input_ids = torch.full((B, max_length), pad_token_id, dtype=dtype_ids, device=device)
        batch_attention_mask = torch.zeros((B, max_length), dtype=attn_mask_list[0].dtype, device=device)
        batch_loss_mask = torch.zeros((B, max_length), dtype=torch.long, device=device)  # usually bool/long

        # target window: [B, length]
        batch_target = torch.empty((B, length), dtype=target_list[0].dtype, device=device)

        mask_tokens = torch.full((length,), mask_token_id, dtype=dtype_ids, device=device)

        # ---- fill each row
        for i in range(B):
            inp = input_ids_list[i]
            att = attn_mask_list[i]
            tgt = target_list[i]

            s = int(starts[i].item())
            # clamp in case tgt shorter than length
            s = min(s, max(tgt.size(0) - length, 0))

            prefix = tgt[:s]  # [s]
            window = tgt[s:s + length]  # [length] (or shorter if tgt too short)

            # if tgt is shorter than length, pad window (rare if your data is valid)
            if window.size(0) < length:
                padded = torch.full((length,), pad_token_id, dtype=tgt.dtype, device=device)
                padded[:window.size(0)] = window
                window = padded

            seq = torch.cat([inp, prefix.to(dtype_ids), mask_tokens], dim=0)  # [seq_len]
            L = seq.size(0)

            batch_input_ids[i, :L] = seq
            batch_attention_mask[i, :L] = 1  # or: torch.cat([att, ones...]) if you need original attn pattern
            batch_loss_mask[i, L - length:L] = 1  # only mask tokens contribute to loss
            batch_target[i] = window

        return {
            "input_ids": batch_input_ids,
            "target": batch_target,                 # [B, length]
            "attention_mask": batch_attention_mask, # [B, max_length]
            "loss_mask": batch_loss_mask,           # [B, max_length]
        }