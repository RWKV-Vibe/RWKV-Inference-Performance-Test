import os, copy, types, gc, sys, re, platform, psutil, time
import numpy as np
from prompt_toolkit import prompt
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
os.environ["RWKV_V7_ON"] = "1"  # enable this for rwkv-7 models
os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "0"  # !!! '1' to compile CUDA kernel (will be faster), requires c++ compiler & cuda libraries !!!

from rwkv.model import RWKV
from rwkv.utils import PIPELINE

########################################################################################################

args = types.SimpleNamespace()
args.strategy = "cuda fp16"  # use CUDA, fp16
args.MODEL_NAME = "C:/models/rwkv7-g1-2.9b-20250519-ctx4096"

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

print("=== System information ===")
print(f"Platform: {platform.platform()}")
print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print("\n")

########################################################################################################
STATE_NAME = None
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

# print(f"Loading model - {args.MODEL_NAME}")
model = RWKV(model=args.MODEL_NAME, strategy=args.strategy)
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")


print_memory_usage("After Loading ")
print("─" * 70)

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
total_tokens_generated = 0  # Only tokens counted for TPS (excluding first tokens)
total_inference_time = 0

while True:
    msg = prompt("User: ")
    msg = msg.strip()
    msg = re.sub(r"\n+", "\n", msg)
    
    if len(msg) > 0:
        # Start timing
        start_time = time.time()
        tokens_this_round = 0
        actual_tokens_for_tps = 0  # Tokens counted for TPS (excluding first token)
        first_token_processed = False
        generation_start_time = None
        
        occurrence = {}
        out_tokens = []
        out_last = 0

        out = run_rnn("User: " + msg + "\n\nAssistant:")
        print("\nAssistant:", end="")

        for i in range(99999):
            for n in occurrence:
                out[n] -= GEN_alpha_presence + occurrence[n] * GEN_alpha_frequency  # repetition penalty
            out[0] -= 1e10  # disable END_OF_TEXT

            token = pipeline.sample_logits(out, temperature=GEN_TEMP, top_p=GEN_TOP_P)
            out, model_state = model.forward([token], model_state)
            model_tokens += [token]
            out_tokens += [token]
            tokens_this_round += 1

            # Start timing from the second token (after first token initialization)
            if not first_token_processed:
                first_token_processed = True
                generation_start_time = time.time()  # Reset timing after first token
            else:
                actual_tokens_for_tps += 1  # Only count tokens after the first one

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
        
        if generation_start_time is not None:
            generation_time = end_time - generation_start_time  # Time from second token onwards
        else:
            generation_time = 0
        
        # Update total statistics (using actual tokens for TPS calculation)
        total_tokens_generated += actual_tokens_for_tps
        total_inference_time += generation_time
        
        # Calculate speed (excluding first token initialization time)
        current_speed = actual_tokens_for_tps / generation_time if generation_time > 0 and actual_tokens_for_tps > 0 else 0
        average_speed = total_tokens_generated / total_inference_time if total_inference_time > 0 else 0

        print("─" * 70)
        print(f"[Current Generation]: {actual_tokens_for_tps} for TPS | Time: {generation_time:.2f}s | Speed: {current_speed:.2f} tokens/s")
        print(f"[Total Statistics]: {total_tokens_generated} tokens | Average Speed: {average_speed:.2f} tokens/s")
        print_memory_usage("Current ")
        
        # Optional: Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # print("GPU cache cleared")
        print(f"{'─' * 70}\n")

    else:
        print("!!! Error: please say something !!!")