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
            "input_ids": [],
            "loss_mask": []
        }
        for i in range(len(examples['id'])):
            messages = [
                {"role": "system",
                 "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
            ]
            convroles = ["user", "assistant"]
            roles = {"human": "user", "gpt": "assistant"}
            source = examples['conversations'][i]
            if not source:
                continue
            if roles[source[0]["from"]] != "user":
                # Skip the first one if it is not from human
                source = source[1:]
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == convroles[j % 2], f"{i}"
                # if sentence["from"]=="gpt":
                #     sentence["value"]=" "+sentence["value"]
                messages.append(
                    {"role": role, "content": sentence["value"]}
                )
            conversation = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id = tokenizer.unk_token_id

            input_ids = tokenizer(
                conversation,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids[0]
            # filtering out the samples which is longer than max_len
            if len(input_ids) > max_len:
                continue
            loss_mask = torch.ones_like(input_ids)
            # print(i)

            sep = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

            total_len = len(input_ids)

            sep2 = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
            turns = conversation.split(sep2)

            turns[1] = turns[0] + sep2 + turns[1]
            turns = turns[1:]

            cur_len = 1
            loss_mask[:cur_len] = 0
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)

                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

                # Ignore the user instructions
                if i == 0:
                    loss_mask[cur_len: cur_len + instruction_len - 2] = 0
                else:
                    loss_mask[cur_len - 3: cur_len + instruction_len + 1] = 0
                cur_len += turn_len
                if i != 0:
                    cur_len += 3
                # cur_len+=2

                # if i != 0 and not tokenizer.legacy:
                #     # The legacy and non-legacy modes handle special tokens differently
                #     cur_len -= 1

            loss_mask[cur_len:] = 0
            attention_mask = torch.ones_like(loss_mask)

            # new_examples["conversation"].append(conversation)
            new_examples["input_ids"].append(input_ids[None, :])
            new_examples["loss_mask"].append(loss_mask[None, :])
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



# def build_dataset_rank(
#         tokenizer, datapath, max_len
# ):

#     # ds = load_dataset('json', data_files=datapath)
#     ds = load_from_disk(datapath)
#     # ds = ds['train']
#     ds = ds.shuffle(seed=42)
#     ds1 = ds
#     original_columns1 = ds1.column_names
#     num_proc = 8

#     def preprocess_function(examples):
#         new_examples = {
#             "attention_mask": [],
#             "input_ids": [],
#             "loss_mask": []
#         }
#         for i in range(len(examples['id'])):
#             messages = [
#                 {"role": "system",
#                  "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
#             ]
#             convroles = ["user", "assistant"]
#             roles = {"human": "user", "gpt": "assistant"}
#             source = examples['conversations'][i]
#             if not source:
#                 continue
#             if roles[source[0]["from"]] != "user":
#                 # Skip the first one if it is not from human
#                 source = source[1:]
#             for j, sentence in enumerate(source):
#                 role = roles[sentence["from"]]
#                 assert role == convroles[j % 2], f"{i}"
#                 # if sentence["from"]=="gpt":
#                 #     sentence["value"]=" "+sentence["value"]
#                 messages.append(
#                     {"role": role, "content": sentence["value"]}
#                 )
#             conversation = tokenizer.apply_chat_template(
#                 messages,
#                 tokenize=False,
#                 add_generation_prompt=False,
#             )

#             if not tokenizer.pad_token_id:
#                 tokenizer.pad_token_id = tokenizer.unk_token_id

#             input_ids = tokenizer(
#                 conversation,
#                 return_tensors="pt",
#                 add_special_tokens=False,
#             ).input_ids[0]
#             # filtering out the samples which is longer than max_len
#             if len(input_ids) > max_len:
#                 continue
#             loss_mask = torch.ones_like(input_ids)
#             # print(i)

#             sep = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

#             total_len = len(input_ids)

#             sep2 = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
#             turns = conversation.split(sep2)

#             turns[1] = turns[0] + sep2 + turns[1]
#             turns = turns[1:]

#             cur_len = 1
#             loss_mask[:cur_len] = 0
#             for i, turn in enumerate(turns):
#                 if turn == "":
#                     break
#                 turn_len = len(tokenizer(turn).input_ids)

#                 parts = turn.split(sep)
#                 if len(parts) != 2:
#                     break
#                 parts[0] += sep
#                 # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
#                 instruction_len = len(tokenizer(parts[0]).input_ids) - 1

#                 # Ignore the user instructions
#                 if i == 0:
#                     loss_mask[cur_len: cur_len + instruction_len - 2] = 0
#                 else:
#                     loss_mask[cur_len - 3: cur_len + instruction_len + 1] = 0
#                 cur_len += turn_len
#                 if i != 0:
#                     cur_len += 3
#                 # cur_len+=2

#                 # if i != 0 and not tokenizer.legacy:
#                 #     # The legacy and non-legacy modes handle special tokens differently
#                 #     cur_len -= 1

#             loss_mask[cur_len:] = 0
#             attention_mask = torch.ones_like(loss_mask)

#             # new_examples["conversation"].append(conversation)
#             new_examples["input_ids"].append(input_ids[None, :])
#             new_examples["loss_mask"].append(loss_mask[None, :])
#             new_examples["attention_mask"].append(attention_mask[None, :])

#         return new_examples

#     ds1 = ds1.map(
#         preprocess_function,
#         batched=True,
#         num_proc=num_proc,
#         remove_columns=original_columns1,
#         load_from_cache_file=False
#     )

#     ds1.set_format(type="torch")
#     return ds1


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

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['input_ids'].shape[1] for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_attention_mask = torch.cat(
            [self.paddingtensor2D(item['attention_mask'], max_length) for item in features])
        batch_loss_mask = torch.cat(
            [self.paddingtensor2D(item['loss_mask'], max_length) for item in features])

        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch

