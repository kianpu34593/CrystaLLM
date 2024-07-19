"""
Adapted from:
https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/train.py
"""
import os
from dataclasses import dataclass
from typing import Union
import math
import time
import wandb

wandb.login()

from crystallm import parse_config
from omegaconf import OmegaConf
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import pickle
from contextlib import nullcontext

from crystallm import (
    GPT,
    GPTConfig,
)


@dataclass
class TrainDefaults:
    out_dir: str = "out"  # the path to the folder where the model checkpoints will be stored
    eval_interval: int = 250  # how often to evaluate against the validation set
    log_interval: int = 1  # how often to print to
    eval_iters_train: int = 200
    eval_iters_val: int = 200
    eval_only: bool = False  # if True, script exits right after the first eval
    always_save_checkpoint: bool = False  # if True, always save a checkpoint after each eval
    init_from: str = "scratch"  # 'scratch' or 'resume'

    # data
    dataset: str = ""  # the path to the folder containing the .bin files with encoded tokens
    gradient_accumulation_steps: int = 40  # used to simulate larger batch sizes
    batch_size: int = 64  # if gradient_accumulation_steps > 1, this is the micro-batch size
    block_size: int = 2048  # context of up to `block_size` previous characters

    # model
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
    bias: bool = False  # do we use bias inside LayerNorm and Linear layers?

    # AdamW optimizer
    learning_rate: float = 6e-4  # max learning rate
    max_iters: int = 600000  # total number of training iterations
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95  # make a bit bigger because number of tokens per iter is small
    grad_clip: float = 1.0  # clip gradients at this value, or disable if == 0.0

    # learning rate decay settings
    decay_lr: bool = True  # whether to decay the learning rate
    warmup_iters: int = 2000  # how many steps to warm up for; not super necessary potentially
    lr_decay_iters: int = 600000  # should be ~= max_iters per Chinchilla
    min_lr: float = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    # DDP settings
    backend: str = 'nccl' # 'nccl', 'gloo', etc.

    # system
    device: str = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype: str = "bfloat16"  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile: bool = True  # use PyTorch 2.0 to compile the model to be faster
    underrep_p: float = 0.0
    validate: bool = False  # whether to evaluate the model using the validation set


def read_start_indices(
    max_start_index: int,
    data_dir: str,
    starts_fname: str,
    on_condition: bool = True,
    required: bool = False,
) -> Union[torch.Tensor, None]:
    start_indices = None
    starts_path = os.path.join(data_dir, starts_fname)
    if on_condition:
        if os.path.exists(starts_path):
            print(f"Reading start indices from {starts_path}...")
            with open(starts_path, "rb") as f:
                start_indices = torch.tensor(pickle.load(f))  # should be sorted
            # remove indices that would result in out-of-bounds sequences
            start_indices = start_indices[start_indices <= max_start_index]
        elif required:
            raise Exception(f"Expected to find a file in dataset dir named '{starts_fname}'")
    return start_indices


if __name__ == "__main__":
    C = parse_config(TrainDefaults)
    run = wandb.init(
        # Set the project where this run will be logged
        project="crystallm-art-prod",
        name=C.out_dir.split('/')[-1],
        # Track hyperparameters and run metadata
        config={
            "epochs": C.max_iters,
        },
    )

    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        init_process_group(backend=C.backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert C.gradient_accumulation_steps % ddp_world_size == 0
        C.gradient_accumulation_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
    
    if master_process:
        print("Using configuration:")
        print(OmegaConf.to_yaml(C))

    tokens_per_iter = C.gradient_accumulation_steps * ddp_world_size * C.batch_size * C.block_size
    if master_process:
        print(f"tokens per iteration will be: {tokens_per_iter:,}")
        print(f"Creating {C.out_dir}...")
        os.makedirs(C.out_dir, exist_ok=True)

    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = "cuda" if "cuda" in C.device else "cpu"  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[C.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    if not C.dataset:
        raise Exception("The 'dataset' option is required and cannot be empty")

    train_data = np.memmap(os.path.join(C.dataset, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(C.dataset, "val.bin"), dtype=np.uint16, mode="r") if C.validate else None

    cif_start_indices = read_start_indices(
        max_start_index=len(train_data) - C.block_size,
        data_dir=C.dataset,
        starts_fname="starts.pkl",
    )

    cif_start_indices_val = read_start_indices(
        max_start_index=(len(val_data) - C.block_size) if C.validate else -1,
        data_dir=C.dataset,
        starts_fname="starts_val.pkl",
        on_condition=C.validate,
    )

    cif_start_indices_underrep = read_start_indices(
        max_start_index=len(train_data) - C.block_size,
        data_dir=C.dataset,
        starts_fname="starts_underrep.pkl",
        on_condition=C.underrep_p > 0,
        required=True,
    )

    def get_batch(split):
        data = train_data if split == "train" else val_data

        ix = torch.randint(len(data) - C.block_size, (C.batch_size,))
        if split == "train":
            if C.underrep_p is not None and np.random.random() < C.underrep_p:
                ix = cif_start_indices_underrep[torch.randperm(len(cif_start_indices_underrep))[:C.batch_size]]
            elif cif_start_indices is not None:
                ix = cif_start_indices[torch.randperm(len(cif_start_indices))[:C.batch_size]]
        elif cif_start_indices_val is not None:
            ix = cif_start_indices_val[torch.randperm(len(cif_start_indices_val))[:C.batch_size]]

        x = torch.stack([torch.from_numpy((data[i:i + C.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + C.block_size]).astype(np.int64)) for i in ix])

        if device_type == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(C.device, non_blocking=True), y.pin_memory().to(C.device, non_blocking=True)
        else:
            x, y = x.to(C.device), y.to(C.device)
        return x, y

    iter_num = 0
    best_val_loss = 1e9

    meta_path = os.path.join(C.dataset, "meta.pkl")
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        meta_vocab_size = meta["vocab_size"]
        if master_process:
            print(f"Found vocab_size = {meta_vocab_size} (inside {meta_path})")

    model_args = dict(n_layer=C.n_layer, n_head=C.n_head, n_embd=C.n_embd, block_size=C.block_size,
                      bias=C.bias, vocab_size=None, dropout=C.dropout)
    if C.init_from == "scratch":
        if master_process:
            print("Initializing a new model from scratch...")
            if meta_vocab_size is None:
                print("Defaulting to vocab_size of 371...")
        model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 371
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif C.init_from == "resume":
        if master_process:
            print(f"Resuming training from {C.out_dir}...")
        ckpt_path = os.path.join(C.out_dir, "latest.pt") #modified from "ckpt.pt"
        checkpoint = torch.load(ckpt_path, map_location=C.device)
        checkpoint_model_args = checkpoint["model_args"]
        # force these config attributes to be equal otherwise we can't even resume training;
        #  the rest of the attributes (e.g. dropout) can stay as desired
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = checkpoint_model_args[k]
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint["model"]
        # fix the keys of the state dictionary
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]

    # crop down the model block size if desired, using model surgery
    if C.block_size < model.config.block_size:
        model.crop_block_size(C.block_size)
        model_args["block_size"] = C.block_size  # so that the checkpoint will have the right value
    model.to(C.device)

    # initialize a GradScaler; if enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(C.dtype == "float16"))

    optimizer = model.configure_optimizers(C.weight_decay, C.learning_rate, (C.beta1, C.beta2))
    if C.init_from == "resume":
        optimizer.load_state_dict(checkpoint["optimizer"])

    if C.compile:
        if master_process:
            print("Compiling the model (takes a ~minute)...")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0

    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split, eval_iters in [("train", C.eval_iters_train), ("val", C.eval_iters_val)]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < C.warmup_iters:
            return C.learning_rate * it / C.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > C.lr_decay_iters:
            return C.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - C.warmup_iters) / (C.lr_decay_iters - C.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return C.min_lr + coeff * (C.learning_rate - C.min_lr)

    # training loop
    X, Y = get_batch("train")
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    model = model.module if ddp else model # unwrap DDP container if needed
    running_mfu = -1.0
    while True:

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if C.decay_lr else C.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % C.eval_interval == 0 and master_process:
            if C.validate:
                losses = estimate_loss()
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                wandb.log({"step": iter_num, "train loss": losses['train'], "val loss": losses['val']})
            if (C.validate and losses["val"] < best_val_loss) or C.always_save_checkpoint:
                best_val_loss = losses["val"] if C.validate else 0.
                if iter_num > 0:
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": dict(C),
                    }
                    print(f"saving checkpoint to {C.out_dir}...")
                    torch.save(checkpoint, os.path.join(C.out_dir, "ckpt.pt"))
            #save the latest checkpoint
            checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": dict(C),
                    }
            print(f"saving latest checkpoint to {C.out_dir}...")
            torch.save(checkpoint, os.path.join(C.out_dir, "latest.pt"))
        if iter_num == 0 and C.eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(C.gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (micro_step == C.gradient_accumulation_steps - 1)
            with ctx:
                logits, loss = model(X, Y)
                loss=loss/C.gradient_accumulation_steps
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch("train")
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if C.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % C.log_interval == 0  and master_process:
            lossf = loss.item() * C.gradient_accumulation_steps # loss as float. note: this is a CPU-GPU sync point
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = model.estimate_mfu(C.batch_size * C.gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > C.max_iters:
            break
            
    if ddp:
        destroy_process_group()
