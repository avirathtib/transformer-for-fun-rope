from tqdm.auto import tqdm
import math
import os
import argparse
import time
from transformers import GPT2TokenizerFast
from datasets import load_from_disk
from torch.utils.data import DataLoader
# from accelerate import Accelerator, DistributedType
from model import GPT
import torch.nn as nn
import torch.nn.functional as F
import torch
from data_processing import MemmapDataset
from data_processing_fineweb import FineWebDataset, FineWebBatchIterator, FineWebDataLoader
from hellaswag_eval import quick_hellaswag_eval
import wandb

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train GPT model with optional checkpoint resumption')
parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                    help='Path to checkpoint file to resume training from')
parser.add_argument('--wandb_project', type=str, default='gpt-training',
                    help='Weights & Biases project name')
parser.add_argument('--wandb_run_name', type=str, default=None,
                    help='Weights & Biases run name')
parser.add_argument('--disable_wandb', action='store_true',
                    help='Disable Weights & Biases logging')
args = parser.parse_args()

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
initial_lr = 6e-4
final_lr = 0.1 * initial_lr
epochs = 4 # with new FineWeb shard data loader, we will switch to step based calculations - can still be used as epoch multiplier
total_steps = 152587 
warmup_steps = 0.06 * total_steps


if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("tokeinzer vocab size", tokenizer.vocab_size)
fineweb_ds = FineWebDataset(folder_path='./edu_fineweb10B')
fineweb_ds_val = FineWebDataset(folder_path='./edu_fineweb10B_val')

train_loader = DataLoader(fineweb_ds, batch_size=16, shuffle=True, pin_memory=True, num_workers=0)
val_loader = DataLoader(fineweb_ds_val, batch_size=16, shuffle=False, pin_memory=True, num_workers=0)

fw_loader = FineWebDataLoader(folder_path='./edu_fineweb10B', batch_size=16, token_length=1024)
steps_per_epoch = 152587  # 1 epoch in your token-counted setup
train_iterator = FineWebBatchIterator(fw_loader, steps_per_epoch)
fw_val_loader = FineWebDataLoader(folder_path='./edu_fineweb10B_val', batch_size=16, token_length=1024)
device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")

# total_steps = int(epochs * len(train_loader))
# warmup_steps = max(1000, int(total_steps * 0.005))

torch.set_float32_matmul_precision('high')


def init_weights(mod):
    std = 0.02
    if (isinstance(mod, nn.Linear)):
        if (hasattr(mod, 'NANOGPT_SCALE_INIT')):
            std *= (2 * 12) ** -0.5
        torch.nn.init.normal_(mod.weight, mean = 0, std = std)
        
        if mod.bias is not None:
            torch.nn.init.zeros_(mod.bias)
    elif (isinstance(mod, nn.Embedding)):
        torch.nn.init.normal_(mod.weight, mean = 0, std = 0.02)        

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # cosine decay
        progress = (current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))  # goes from 1 -> 0
        return max(final_lr / initial_lr, cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

model = GPT(50304, 128, 128 * 3, 2, 4, 0.1).to(device)
model.apply(init_weights)
model = torch.compile(model)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
non_trainable_params = total_params - trainable_params
print(f"Trainable params: {trainable_params:,}")
print(f"Non-trainable params: {non_trainable_params:,}")
print(f"Total params: {total_params:,}")
total = sum(p.numel() for p in model.parameters())
emb = sum(p.numel() for n,p in model.named_parameters() if 'embedding' in n.lower() and 'embed' in n.lower())
lm_head = sum(p.numel() for n,p in model.named_parameters() if n.startswith('fc') or 'fc' in n.lower() and 'weight' in n.lower() and p.dim()==2)
print("total params:", total)
print("embedding params:", emb)
print("lm_head params:", lm_head)

# params = [(n, p.shape, p.numel()) for n,p in model.named_parameters()]
# params_sorted = sorted(params, key=lambda x: x[2], reverse=True)
# for n, shape, numel in params_sorted[:40]:
#     print(f"{n:40s} shape={shape!s:20s}  numel={numel:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, betas = (0.9, 0.95), eps=1e-8)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
criterion = nn.CrossEntropyLoss(ignore_index=-100)

# Initialize Weights & Biases
if not args.disable_wandb:
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "model_params": total_params,
            "trainable_params": trainable_params,
            "vocab_size": 50304,
            "embed_dim": 768,
            "n_layers": 12,
            "n_heads": 12,
            "dropout": 0.1,
            "initial_lr": initial_lr,
            "final_lr": final_lr,
            "epochs": epochs,
            "batch_size": 16,
            "warmup_steps": warmup_steps,
            "total_steps": total_steps
        }
    )
else:
    wandb = None

# model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)


def save_checkpoint(model, optimizer, scheduler, step, epoch, loss, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': step,
        'epoch': epoch,
        'loss': loss
    }
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{step}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    step = checkpoint['step']
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print(f"Resumed from step {step}, epoch {epoch}, loss {loss:.4f}")
    return step, epoch, loss

def evaluate_model_fw(model, fw_val_loader, criterion, device, val_steps=None, max_tokens=None):
    """
    fw_val_loader: FineWebDataLoader instance
    val_steps: number of batches to evaluate (if None, go until we've consumed all tokens once)
    max_tokens: optional cap on tokens evaluated (int)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    pad_ignore = criterion.ignore_index

    # If fw_val_loader is stateful, reset to start for deterministic eval:
    if hasattr(fw_val_loader, "current_file_index"):
        fw_val_loader.current_file_index = 0
        fw_val_loader.current_token_index = 0
        fw_val_loader._load_file()

    steps_done = 0
    seen_tokens = 0

    with torch.no_grad():
        while True:
            if val_steps is not None and steps_done >= val_steps:
                break

            # fetch next batch (x,y) as torch tensors on CPU
            x, y = fw_val_loader.next_batch()  
            x = x.to(device)
            y = y.to(device)

            logits = model(x)                    # (B, S, V)
            V = logits.size(-1)

            labels_flat = y.view(-1)
            labels_flat[labels_flat == tokenizer.pad_token_id] = -100

            loss = criterion(logits.view(-1, V), labels_flat)
            valid_tokens = int((labels_flat != -100).sum().item())

            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens
            seen_tokens += valid_tokens

            steps_done += 1

            if max_tokens is not None and seen_tokens >= max_tokens:
                break

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = math.exp(avg_loss) if avg_loss < 1000 else float("inf")
    model.train()
    return avg_loss, perplexity, seen_tokens, steps_done


model.train()
step = 0
start_epoch = 0
lr_history = []

# Load checkpoint if specified
if args.resume_from_checkpoint:
    step, start_epoch, _ = load_checkpoint(args.resume_from_checkpoint, model, optimizer, scheduler, device)
    print(f"Resuming training from step {step}, epoch {start_epoch}")

eval_every = 1500
eval_max_batches = 200
hellaswag_eval_every = 7500  # Run HellaSwag evaluation every 500 steps
hellaswag_samples = 50     # Number of HellaSwag samples to evaluate

for epoch in range(start_epoch, epochs):
    # outer tqdm: epoch-level
    epoch_iterator = tqdm(train_iterator, total=steps_per_epoch, desc=f"Epoch {epoch+1}/{epochs}")

    for batch_idx, (inputs, targets) in enumerate(epoch_iterator):
        # Start timing for tokens/sec calculation
        step_start_time = time.perf_counter()

        # print(batch)
        inputs = inputs.to(device)
        targets = targets.to(device)


        # show a few input->target token pairs decoded
        for i in range(3):
            inp = inputs[i,:10].tolist()
            tgt = targets[i,:10].tolist()
            
            break

        labels_flat = targets.contiguous().view(-1)
        

        # with torch.autocast(device_type = device.type, dtype = torch.bfloat16):
        logits = model(inputs)               # (B, S, V)
        V = logits.size(-1)
        labels_flat = targets.contiguous().view(-1)
        labels_flat[labels_flat == tokenizer.pad_token_id] = -100
        loss = criterion(logits.view(-1, V), labels_flat)
        valid_tokens = int((labels_flat != -100).sum().item())                   

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        # torch.cuda.synchronize()

        # Calculate tokens per second
        step_end_time = time.perf_counter()
        step_duration = step_end_time - step_start_time
        batch_size, token_length = inputs.shape
        total_tokens = batch_size * token_length
        tokens_per_sec = total_tokens / step_duration


    
        V = logits.size(-1)



        current_lr = scheduler.get_last_lr()[0]
        lr_history.append(current_lr)
        step += 1

        phase = "Warmup" if step < warmup_steps else "Decay"
        # update tqdm bar (loss, lr, phase, tokens/sec)
        epoch_iterator.set_postfix({
            "step": step,
            "phase": phase,
            "loss": f"{loss.item():.4f}",
            "lr": f"{current_lr:.2e}",
            "tok/s": f"{tokens_per_sec:.0f}"
        })

        val_total_tokens = 100000000   # fill in actual number (tokens)
        val_steps = max(1, int(val_total_tokens / (fw_val_loader.batch_size * fw_val_loader.token_length)))

        # inside training loop where you evaluate:
        if step % eval_every == 0 and step > 0:
            # deterministic eval: reset loader inside evaluate_model_fw() (function does that)
            val_loss, val_ppl, seen_tokens, seen_batches = evaluate_model_fw(
                model, fw_val_loader, criterion, device, val_steps=val_steps
            )
            tqdm.write(f"[Validation] Step {step}: Batches={seen_batches}, Tokens={seen_tokens}, Loss={val_loss:.4f}, PPL={val_ppl:.2f}")
            save_checkpoint(model, optimizer, scheduler, step, epoch, val_loss)
            # HellaSwag evaluation
            hellaswag_metrics = None
            if step % hellaswag_eval_every == 0:
                tqdm.write(f"[HellaSwag] Running evaluation on {hellaswag_samples} samples...")
                hellaswag_metrics = quick_hellaswag_eval(model, tokenizer, device, hellaswag_samples)
                tqdm.write(f"[HellaSwag] Step {step}: Accuracy={hellaswag_metrics['accuracy']:.4f} ({hellaswag_metrics['correct']}/{hellaswag_metrics['total']})")

            # Log to wandb
            wandb_log = {
                "train/step": step,
                "train/epoch": epoch,
                "train/loss": loss.item(),
                "train/learning_rate": current_lr,
                "train/tokens_per_sec": tokens_per_sec,
                "val/loss": val_loss,
                "val/perplexity": val_ppl,
                "val/tokens": seen_tokens,
                "val/batches": seen_batches
            }

            if hellaswag_metrics is not None:
                wandb_log.update({
                    "hellaswag/accuracy": hellaswag_metrics['accuracy'],
                    "hellaswag/correct": hellaswag_metrics['correct'],
                    "hellaswag/total": hellaswag_metrics['total']
                })

            if wandb is not None:
                wandb.log(wandb_log)



    
        # if batch_idx >= 1000:
        #     break

    # end-of-epoch quick validation
    val_loss, val_ppl, seen_tokens, seen_batches = evaluate_model_fw(
                model, fw_val_loader, criterion, device, val_steps=val_steps
            )
    print(f'End of Epoch {epoch+1} - Validation Batches: {seen_batches}, Tokens: {seen_tokens}, Loss: {val_loss:.4f}, Perplexity: {val_ppl:.2f}')

    # End-of-epoch HellaSwag evaluation
    print(f"Running HellaSwag evaluation at end of epoch {epoch+1}...")
    hellaswag_metrics = quick_hellaswag_eval(model, tokenizer, device, hellaswag_samples)
    print(f'End of Epoch {epoch+1} - HellaSwag Accuracy: {hellaswag_metrics["accuracy"]:.4f} ({hellaswag_metrics["correct"]}/{hellaswag_metrics["total"]})')

    # Log end-of-epoch metrics to wandb
    if wandb is not None:
        wandb.log({
            "epoch": epoch + 1,
            "epoch_val/loss": val_loss,
            "epoch_val/perplexity": val_ppl,
            "epoch_hellaswag/accuracy": hellaswag_metrics['accuracy'],
            "epoch_hellaswag/correct": hellaswag_metrics['correct'],
            "epoch_hellaswag/total": hellaswag_metrics['total']
        })

# Finish wandb run
if wandb is not None:
    wandb.finish()
