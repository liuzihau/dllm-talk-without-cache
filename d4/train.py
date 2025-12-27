import os, re, json
import argparse
import math
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

def manual_init_talking_ml(model, config):
    print("Applying manual initialization to talking_ml...")
    init_std = getattr(config, 'init_std', 0.02)
    
    # 1. Initialize the FC layer (Critical Fix)
    if hasattr(model.talking_ml, 'fc') and isinstance(model.talking_ml.fc, nn.Linear):
        print("Initializing talking_ml.fc ...")
        nn.init.normal_(model.talking_ml.fc.weight, std=init_std)
        if model.talking_ml.fc.bias is not None:
            nn.init.zeros_(model.talking_ml.fc.bias)

    # 2. Initialize Transformer Blocks with Depth Scaling
    # This prevents gradient explosion in deep layers
    blocks = model.talking_ml.transformer.blocks
    for layer_idx, block in enumerate(blocks):
        scaled_std = init_std / math.sqrt(2 * (layer_idx + 1))
        for name, module in block.named_modules():
            if isinstance(module, nn.Linear):
                # Output layers get scaled init
                if "attn_out" in name or "ff_out" in name:
                    nn.init.trunc_normal_(module.weight, mean=0.0, std=scaled_std, a=-3*scaled_std, b=3*scaled_std)
                # Projections get standard init
                elif "proj" in name: # q_proj, k_proj, etc.
                    nn.init.trunc_normal_(module.weight, mean=0.0, std=init_std, a=-3*init_std, b=3*init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

def print_param_summary(model):
    total = trainable = 0
    for n, p in model.named_parameters():
        num = p.numel()
        total += num
        if p.requires_grad:
            trainable += num
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Frozen params:    {total - trainable:,}")

def freeze_all_but_talking_ml(model: torch.nn.Module):
    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze talking_ml only
    for p in model.talking_ml.parameters():
        p.requires_grad = True

    # Optional: sanity prints
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}")
    print(f"Trainable (talking_ml) params: {trainable:,}")

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
    if generator is not None:
        # Ensure generator is on the same device
        assert generator.device == device
        scores = torch.rand((B, C), device=device, generator=generator)
    else:
        scores = torch.rand((B, C), device=device)
    scores = scores.masked_fill(~active, float("-inf"))

    # pick up to k active positions
    idx = scores.topk(k=min(k, C), dim=1).indices    # [B, k]

    # If some rows have <k active, topk will include -inf positions; filter them out:
    chosen_active = active.gather(1, idx)            # [B, k] bool

    rows = torch.arange(B, device=device).unsqueeze(1).expand_as(idx)  # [B, k]

    # 1. Clone input_ids so we don't modify the version used by the previous forward pass
    input_ids = input_ids.clone() 
    
    # 2. Clone loss_mask if it is also part of the graph (though usually it's just a buffer)
    loss_mask = loss_mask.clone()

    rows = rows[chosen_active]
    cols = idx[chosen_active]

    # Now this modification is safe because it happens on the clone
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
freeze_all_but_talking_ml(model)
manual_init_talking_ml(model, model.talking_ml.config)
model.model.eval()
model.talking_ml.train()
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
        
        model_engine.zero_grad()
        with torch.no_grad():
            thought_outputs = model.run_thought_model(
                input_ids=data["input_ids"].to(rank),
                attention_mask=data["attention_mask"].to(rank),
                use_cache=False,
                output_hidden_states=True
            )
            thought_rps = thought_outputs.hidden_states  # [B, S + C, H]
            if torch.isnan(thought_outputs.hidden_states).any():
                print("CRITICAL: Thought model outputs NaN! Check base model loading.")
                raise Exception("force leave")
            else:
                print("Thought model seems healthy.")
            B = thought_rps.size(0)
            H = thought_rps.size(-1)
            thought_rps = thought_rps[mask_bool].view(B, -1, H)
        

        cache_hidden = [[], []]
        plosses = []
        acces = []
        

        input_ids = data['input_ids'][mask_bool].view(data['input_ids'].size(0), -1)
        rps = thought_rps
        talk_attention_mask = torch.ones_like(input_ids, dtype=data["attention_mask"].dtype, device=input_ids.device)
        loss_mask = torch.ones_like(data["target"])
        for idx in range(model.length):

            talk_outputs = model_engine(
                input_ids=input_ids.to(rank),
                inputs_repres=rps,
                attention_mask=talk_attention_mask.to(rank),
                use_cache=False,
                output_hidden_states=True)
            
            logits = talk_outputs.logits.float()
            rps = talk_outputs.hidden_states
            out_logp = F.log_softmax(logits, dim=-1)

            # calculate_ploss
            # V = logits.size(-1)
            # data["target"] = data["target"].to(rank)
            # loss_mask = loss_mask.to(rank)
            # target_p = F.one_hot(data["target"], num_classes=V).float()
            # plogp = target_p * out_logp
            # sum_logit = torch.sum(plogp, 2) * loss_mask
            # loss = -sum_logit.mean() 
            # plosses.append(loss)
            # acces.append(((logits.argmax(-1) == target_p.argmax(-1)) * loss_mask.squeeze(-1)).sum().item() / (loss_mask.sum().item() + 1e-6))
            
            # target = data["target"].to(out_logp.device)
            # mask = loss_mask.float().to(out_logp.device)  # [B, L]
            # loss = calculate_ploss(out_logp, target, mask)
            # acc = calculate_acc(logits, target, mask)

            # plosses.append(loss)
            # acces.append(acc.item())
            

            # --- CORRECTED LOSS CALCULATION START ---
            V = logits.size(-1)
            data["target"] = data["target"].to(rank)
            loss_mask = loss_mask.to(rank)

            # Instead of one-hot multiplication, gather the log-prob of the correct indices
            # data["target"] shape: [B, L] -> unsqueeze to [B, L, 1] for gather
            target_indices = data["target"].unsqueeze(-1).long()

            # Gather values along the vocab dimension (dim=2)
            # resulting shape: [B, L, 1] -> squeeze back to [B, L]
            target_logp = out_logp.gather(2, target_indices).squeeze(2)

            # Calculate loss only on valid tokens
            # We use negative log likelihood: -log(p)
            mask = loss_mask.float()
            loss = -(target_logp * mask).sum() / (mask.sum().clamp_min(1e-6))

            # Note: If loss_mask can be all zeros, mean() might be weak. 
            # You might prefer: loss = -(masked_logp.sum() / (loss_mask.sum() + 1e-6))
            # But keeping consistent with your snippet:
            # --- CORRECTED LOSS CALCULATION END ---

            plosses.append(loss)

            # Calculate Accuracy (for logging)
            pred = logits.argmax(dim=-1)
            correct = (pred == data["target"]) & loss_mask.bool()
            acc = correct.float().sum() / (loss_mask.sum() + 1e-6)
            acces.append(acc.item())


            input_ids, loss_mask = denoise_k_step(input_ids.to(rank), data["target"], loss_mask)
           

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