import os, copy, types, gc, sys, re
import numpy as np
from prompt_toolkit import prompt
import torch
import time
import psutil

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
os.environ["RWKV_V7_ON"] = "1"  # enable this for rwkv-7 models
os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "1"  # !!! '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries !!!

from rwkv.model import RWKV
from rwkv.utils import PIPELINE

########################################################################################################

args = types.SimpleNamespace()
args.strategy = "cuda fp16"  # use CUDA, fp16
args.MODEL_NAME = "/home/rwkv/Example/models/rwkv7-g1-2.9b-20250519-ctx4096"

# Add monitoring functions
def get_memory_usage():
    """Get memory usage information"""
    memory_info = {}
    
    # GPU Memory
    if torch.cuda.is_available():
        memory_info['vram_used'] = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_info['vram_cached'] = torch.cuda.memory_reserved() / 1024**3  # GB
        memory_info['vram_total'] = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        memory_info['vram_percent'] = (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100
    else:
        memory_info['vram_used'] = 0
        memory_info['vram_cached'] = 0
        memory_info['vram_total'] = 0
        memory_info['vram_percent'] = 0
    
    return memory_info

def print_memory_usage(prefix=""):
    """Print VRAM usage information"""
    mem = get_memory_usage()
    if torch.cuda.is_available():
        print(f"[{prefix}VRAM Usage]: {mem['vram_used']:.2f}GB/{mem['vram_total']:.2f}GB ({mem['vram_percent']:.1f}%) | Cache: {mem['vram_cached']:.2f}GB")
    else:
        print(f"{prefix}No GPU detected")

print("\n")
print_memory_usage("Before Loading ")

########################################################################################################
STATE_NAME = None  # use vanilla zero initial state?

# use custom state? much better chat results (download from https://huggingface.co/BlinkDL/temp-latest-training-models/tree/main)
# note: this is English Single-round QA state (will forget what you previously say)
# STATE_NAME = "E://RWKV-Runner//models//rwkv-x060-eng_single_round_qa-1B6-20240516-ctx2048"

########################################################################################################

GEN_TEMP = 1.0
GEN_TOP_P = 0.3
GEN_alpha_presence = 0.5
GEN_alpha_frequency = 0.5
GEN_penalty_decay = 0.996

if STATE_NAME != None:
    GEN_TOP_P = 0.2
    GEN_alpha_presence = 0.3
    GEN_alpha_frequency = 0.3

CHUNK_LEN = 256  # split input into chunks to save VRAM (shorter -> slower, but saves VRAM)

########################################################################################################

print(f"Loading model - {args.MODEL_NAME}")
model = RWKV(model=args.MODEL_NAME, strategy=args.strategy)
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

print("\n")
print_memory_usage("After Loading ")

model_tokens = []
model_state = None

if STATE_NAME != None:  # load custom state
    args = model.args
    state_raw = torch.load(STATE_NAME + '.pth')
    state_init = [None for i in range(args.n_layer * 3)]
    for i in range(args.n_layer):
        dd = model.strategy[i]
        dev = dd.device
        atype = dd.atype    
        state_init[i*3+0] = torch.zeros(args.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()
        state_init[i*3+1] = state_raw[f'blocks.{i}.att.time_state'].transpose(1,2).to(dtype=torch.float, device=dev).requires_grad_(False).contiguous()
        state_init[i*3+2] = torch.zeros(args.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()
    model_state = copy.deepcopy(state_init)

def run_rnn(ctx):
    global model_tokens, model_state
    ctx = ctx.replace("\r\n", "\n")
    tokens = pipeline.encode(ctx)
    tokens = [int(x) for x in tokens]
    model_tokens += tokens

    while len(tokens) > 0:
        out, model_state = model.forward(tokens[:CHUNK_LEN], model_state)
        tokens = tokens[CHUNK_LEN:]

    return out

if STATE_NAME == None:  # use initial prompt if we are not loading a state
    init_ctx = "User: hi" + "\n\n"
    init_ctx += "Assistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it." + "\n\n"
    run_rnn(init_ctx)
    print(init_ctx, end="")

# Add performance statistics variables
total_tokens_generated = 0
total_inference_time = 0

while True:
    msg = prompt("User: ")
    msg = msg.strip()
    msg = re.sub(r"\n+", "\n", msg)
    
    if len(msg) > 0:
        # Start timing
        start_time = time.time()
        generation_start_time = time.time()
        tokens_this_round = 0
        
        occurrence = {}
        out_tokens = []
        out_last = 0

        out = run_rnn("User: " + msg + "\n\nAssistant:")
        print("\nAssistant:", end="")

        # Record generation start time
        generation_start_time = time.time()

        for i in range(99999):
            for n in occurrence:
                out[n] -= GEN_alpha_presence + occurrence[n] * GEN_alpha_frequency  # repetition penalty
            out[0] -= 1e10  # disable END_OF_TEXT

            token = pipeline.sample_logits(out, temperature=GEN_TEMP, top_p=GEN_TOP_P)
            out, model_state = model.forward([token], model_state)
            model_tokens += [token]
            out_tokens += [token]
            tokens_this_round += 1

            for xxx in occurrence:
                occurrence[xxx] *= GEN_penalty_decay
            occurrence[token] = 1 + (occurrence[token] if token in occurrence else 0)

            tmp = pipeline.decode(out_tokens[out_last:])
            if ("\ufffd" not in tmp) and (not tmp.endswith("\n")):
                print(tmp, end="", flush=True)
                out_last = i + 1

            if "\n\n" in tmp:
                print(tmp, end="", flush=True)
                break

        # Calculate performance metrics
        end_time = time.time()
        total_time = end_time - start_time
        generation_time = end_time - generation_start_time
        
        # Update total statistics
        total_tokens_generated += tokens_this_round
        total_inference_time += generation_time
        
        # Calculate speed
        current_speed = tokens_this_round / generation_time if generation_time > 0 else 0
        average_speed = total_tokens_generated / total_inference_time if total_inference_time > 0 else 0

        print(f"\n" + "─" * 60)
        print(f"[Current Generation]: {tokens_this_round} tokens | Time: {generation_time:.2f}s | Speed: {current_speed:.2f} tokens/s")
        print(f"[Total Statistics]: {total_tokens_generated} tokens | Average Speed: {average_speed:.2f} tokens/s")
        print_memory_usage("Current ")
        
        # Optional: Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU cache cleared")
        print("─" * 60)

    else:
        print("!!! Error: please say something !!!") 