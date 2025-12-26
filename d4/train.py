import os, re, json
import argparse
from dataclasses import dataclass, field
from safetensors import safe_open
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedTokenizerBase, get_linear_schedule_with_warmup
# import accelerate
from accelerate.utils import set_seed
import deepspeed
from tqdm import tqdm

from model.modeling_d4 import D4ModelLM
from training.data_process import build_dataset_rank, DataCollatorWithPadding
from training.utils import AttrDict

# calculate loss work only when one-hot target
def calculate_ploss(out_logp, target, mask):
    denom = mask.sum().clamp_min(1e-6)
    plogp = out_logp.gather(-1, target.long().unsqueeze(-1)).squeeze(-1)  # [B, L]
    return -(plogp * mask).sum() / denom

def calculate_acc(logits, target, mask):
    denom = mask.sum().clamp_min(1e-6)
    pred = logits.argmax(dim=-1)  # [B, L]
    return ((pred == target.long()).float() * mask).sum() / denom

def find_max_state_with_file(directory, filename="zero_to_fp32.py"):
    max_a = -1
    for subdir in os.listdir(directory):
        match = re.match(r"state_(\d+)", subdir)
        if match:
            a_value = int(match.group(1))
            subdir_path = os.path.join(directory, subdir)
            file_path = os.path.join(subdir_path, filename)
            if os.path.isdir(subdir_path) and os.path.exists(file_path):
                max_a = max(max_a, a_value)
    if max_a == -1:
        return None, 0
    return f"{directory}/state_{max_a}", max_a + 1

def denoise_k_step(input_ids, target, loss_mask, k=1, generator=None):
    device = input_ids.device
    B, C = input_ids.shape

    active = loss_mask.bool()
    scores = torch.rand((B, C), device=device, generator=generator)
    scores = scores.masked_fill(~active, float("-inf"))

    # pick up to k active positions
    idx = scores.topk(k=min(k, C), dim=1).indices    # [B, k]

    # If some rows have <k active, topk will include -inf positions; filter them out:
    chosen_active = active.gather(1, idx)            # [B, k] bool

    rows = torch.arange(B, device=device).unsqueeze(1).expand_as(idx)  # [B, k]

    rows = rows[chosen_active]
    cols = idx[chosen_active]

    input_ids[rows, cols] = target[rows, cols]
    loss_mask[rows, cols] = 0
    return input_ids, loss_mask

# Args 
parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--trainpath', type=str,
                    default="nvidia/Llama-Nemotron-Post-Training-Dataset")
parser.add_argument('--testpath', type=str,
                    default="nvidia/Llama-Nemotron-Post-Training-Dataset")
parser.add_argument('--savedir', type=str, default='0')
parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

# Config
deepspeed_config = args.deepspeed_config
with open(deepspeed_config) as f:
    ds_config = json.load(f)
train_config = AttrDict(ds_config["training_parameters"])

# Env related
torch.backends.cuda.matmul.allow_tf32 = True
set_seed(0)


# Model / Tokenizer 
tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct')
model = D4ModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16)
criterion = nn.SmoothL1Loss(reduction="none")
num_epochs = train_config["num_epochs"]
model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                                     model=model,
                                                     model_parameters=model.talking_ml.parameters(),
                                                     )

global_rank = deepspeed.comm.get_rank()
rank = deepspeed.comm.get_local_rank()
world_size = deepspeed.comm.get_world_size()

# Data
traindataset = build_dataset_rank(tokenizer, args.trainpath, train_config['max_len'])
testdataset = build_dataset_rank(tokenizer, args.testpath, train_config['max_len'])

sampler = DistributedSampler(testdataset, num_replicas=world_size, rank=global_rank, shuffle=False)
test_loader = DataLoader(testdataset, batch_size=train_config["bs"], sampler=sampler, num_workers=4, pin_memory=True,
                         collate_fn=DataCollatorWithPadding())

train_sampler = DistributedSampler(traindataset, num_replicas=world_size, rank=global_rank, shuffle=True)
train_loader = DataLoader(traindataset, batch_size=train_config["bs"], sampler=train_sampler, num_workers=4,
                          pin_memory=True,
                          collate_fn=DataCollatorWithPadding())

# Logging / checkpoints
# if global_rank == 0:
#     import wandb

#     wandb.login(key="671d7c1cf894df27e934d661945640534bbc5bd4")
#     wandb.init(project="TalkingMachine", name="2-decoder-layers", config=ds_config)

os.makedirs(args.savedir, exist_ok=True)

checkpoint_path, start_epoch = find_max_state_with_file(args.savedir)
if checkpoint_path:
    print(f"load from {checkpoint_path}")
    model_engine.load_checkpoint(checkpoint_path)

# Assume data structure
"""
data['input_ids']: [B, S + C]
data['target']: [B, C]
data['attention_mask]': [B, S + C]
data['loss_mask']: [B, S + C]
"""

for epoch in range(start_epoch, num_epochs):
    # train_sampler.set_epoch(epoch+1)
    print(f"Now training epoch {epoch}")

    model.talking_ml.train()
    epoch_acces = [[] for _ in range(model.length)]
    epoch_plosses = [[] for _ in range(model.length)]
    for batch_idx, data in enumerate(tqdm(train_loader)):
        mask_bool = data['loss_mask'].bool()
        
        model.zero_grad()
        with torch.no_grad():
            thought_outputs = model.run_thought_model(
                input_ids=data["input_ids"].to(rank),
                attention_mask=data["attention_mask"].to(rank),
                use_cache=False,
                output_hidden_states=True
            )
            thought_rps = thought_outputs.hidden_states  # [B, S + C, H]
            B = thought_rps.size(0)
            H = thought_rps.size(-1)
            thought_rps = thought_rps[mask_bool].view(B, -1, H)
        
        cache_hidden = [[], []]
        plosses = []
        acces = []
        loss_mask = torch.ones_like(data["target"])

        input_ids = data['input_ids'][mask_bool].view(data['input_ids'].size(0), -1)
        rps = thought_rps
        for idx in range(model.length):
        
            talk_outputs = model_engine(
                input_ids=input_ids.to(rank),
                inputs_repres=rps,
                attention_mask=data["attention_mask"].to(rank),
                use_cache=False,
                output_hidden_states=True)
            
            logits = talk_outputs.logits
            rps = talk_outputs.hidden_states
            out_logp = nn.LogSoftmax(dim=2)(logits)

            """ calculate_ploss
            V = logits.size(-1)
            data["target"] = data["target"].to(out_logp.device)
            target_p = F.one_hot(data["target"], num_classes=V).float()
            plogp = target_p * out_logp
            sum_logit = torch.sum(loss_mask * plogp, 2)
            loss = -sum_logit.mean() 
            plosses.append(loss)
            acces.append(((logits.argmax(-1) == target_p.argmax(-1)) * loss_mask.squeeze(-1)).sum().item() / (loss_mask.sum().item() + 1e-6))
            """
            target = data["target"].to(out_logp.device)
            mask = loss_mask.float().to(out_logp.device)  # [B, L]
            loss = calculate_ploss(out_logp, target, mask)
            acc = calculate_acc(logits, target, mask)

            plosses.append(loss)
            acces.append(acc.item())
            input_ids, loss_mask = denoise_k_step(input_ids, target, loss_mask)
           

        ploss_weight = [0.99 ** i for i in range(len(plosses))]
        ploss = sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
        loss = ploss
        print(f"before scale:{plosses}, after scale: {loss}")
        model_engine.backward(loss)

        model_engine.step()

        # if global_rank == 0:
        #     logdict = {"train/lr": optimizer.optimizer.param_groups[0]["lr"]}
        #     for i in range(len(plosses)):
        #         logdict[f"train/ploss_{i}"] = plosses[i].item()
        #     for i in range(len(acces)):
        #         logdict[f"train/acc_{i}"] = acces[i]
        #     wandb.log(logdict)
        epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
        epoch_plosses = [epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))]


    for i in range(len(epoch_acces)):
        acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
        deepspeed.comm.all_reduce(acc_i, op=deepspeed.comm.ReduceOp.AVG)
        acc_i = acc_i.item()
        # if global_rank == 0:
        #     wandb.log({f"train/epochacc_{i}": acc_i})
        #     print(f"Train Epoch [{epoch + 1}/{num_epochs}], position {i},  Acc: {acc_i:.2f}")

    for i in range(len(epoch_plosses)):
        loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
        deepspeed.comm.all_reduce(loss_i, op=deepspeed.comm.ReduceOp.AVG)
        loss_i = loss_i.item()
        # if global_rank == 0:
        #     wandb.log({f"train/epochploss_{i}": loss_i})
        #     print(f"Train Epoch [{epoch + 1}/{num_epochs}], position {i}, pLoss: {loss_i:.2f}")

    epoch_acces = [[] for _ in range(model.length)]
    epoch_plosses = [[] for _ in range(model.length)]

    for batch_idx, data in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            plosses, acces = model_engine(input_ids=data["input_ids"].to(rank),
                                                   attention_mask=data["attention_mask"].to(rank),
                                                   loss_mask=data["loss_mask"],
                                                   )
            epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
            epoch_plosses = [epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))]

    for i in range(len(epoch_acces)):
        acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
        deepspeed.comm.all_reduce(acc_i, op=deepspeed.comm.ReduceOp.AVG)
        acc_i = acc_i.item()
        # if global_rank == 0:
        #     wandb.log({f"test/epochacc_{i}": acc_i})
        #     print(f"Test Epoch [{epoch + 1}/{num_epochs}], position {i},  Acc: {acc_i:.2f}")

    for i in range(len(epoch_plosses)):
        loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
        deepspeed.comm.all_reduce(loss_i, op=deepspeed.comm.ReduceOp.AVG)
        loss_i = loss_i.item()
        # if global_rank == 0:
        #     wandb.log({f"test/epochploss_{i}": loss_i})
        #     print(f"Test Epoch [{epoch + 1}/{num_epochs}], position {i}, pLoss: {loss_i:.2f}")
    # clear out the redundance cahce after each step
    torch.cuda.empty_cache()

    model_engine.save_16bit_model(f"{args.savedir}/state_{epoch}", exclude_frozen_parameters=True)
    if epoch % 10 == 0:
        deepspeed.DeepSpeedEngine.save_checkpoint(model_engine, save_dir=f"{args.savedir}/state_{epoch}")