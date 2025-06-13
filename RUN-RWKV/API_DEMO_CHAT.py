import os, copy, types, gc, sys, re
import numpy as np
from prompt_toolkit import prompt
import torch
import time
import psutil

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
os.environ["RWKV_V7_ON"] = "1" # enable this for rwkv-7 models
os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "0"  # !!! '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries !!!

from rwkv.model import RWKV
from rwkv.utils import PIPELINE

########################################################################################################

args = types.SimpleNamespace()

args.strategy = "cpu fp16"  # use CUDA, fp16

args.MODEL_NAME = "./rwkv7-g1-0.1b-20250307-ctx4096"

########################################################################################################
# 添加监控函数
def get_memory_usage():
    """获取内存和显存使用情况"""
    memory_info = {}
    
    # 系统内存
    memory_info['ram_used'] = psutil.virtual_memory().used / 1024**3  # GB
    memory_info['ram_total'] = psutil.virtual_memory().total / 1024**3  # GB
    memory_info['ram_percent'] = psutil.virtual_memory().percent
    
    # GPU显存
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
    """打印内存使用情况"""
    mem = get_memory_usage()
    print(f"{prefix}内存使用: {mem['ram_used']:.2f}GB/{mem['ram_total']:.2f}GB ({mem['ram_percent']:.1f}%)")
    if torch.cuda.is_available():
        print(f"{prefix}显存使用: {mem['vram_used']:.2f}GB/{mem['vram_total']:.2f}GB ({mem['vram_percent']:.1f}%) | 缓存: {mem['vram_cached']:.2f}GB")
    else:
        print(f"{prefix}未检测到GPU")

########################################################################################################

STATE_NAME = None # use vanilla zero initial state?

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
print_memory_usage("[加载前] ")

model = RWKV(model=args.MODEL_NAME, strategy=args.strategy)
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

print_memory_usage("[加载后] ")

model_tokens = []
model_state = None

if STATE_NAME != None: # load custom state
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

    # print(f"### model ###\n{model_tokens}\n[{pipeline.decode(model_tokens)}]")  # debug

    while len(tokens) > 0:
        out, model_state = model.forward(tokens[:CHUNK_LEN], model_state)
        tokens = tokens[CHUNK_LEN:]

    return out

if STATE_NAME == None: # use initial prompt if we are not loading a state
    init_ctx = "User: hi" + "\n\n"
    init_ctx += "Assistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it." + "\n\n"
    run_rnn(init_ctx)
    print(init_ctx, end="")

print_memory_usage("[初始化后] ")
print("=" * 60)

# 添加性能统计变量
total_tokens_generated = 0
total_inference_time = 0

while True:
    msg = prompt("User: ")
    msg = msg.strip()
    msg = re.sub(r"\n+", "\n", msg)
    if len(msg) > 0:
        # 开始计时
        start_time = time.time()
        generation_start_time = time.time()
        
        occurrence = {}
        out_tokens = []
        out_last = 0
        tokens_this_round = 0

        out = run_rnn("User: " + msg + "\n\nAssistant:")
        print("\nAssistant:", end="")

        # 记录生成开始时间
        generation_start_time = time.time()

        for i in range(99999):
            for n in occurrence:
                out[n] -= GEN_alpha_presence + occurrence[n] * GEN_alpha_frequency # repetition penalty
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
            if ("\ufffd" not in tmp) and (not tmp.endswith("\n")):  # only print & update out_last when it's a valid utf-8 string and not ending with \n
                print(tmp, end="", flush=True)
                out_last = i + 1

            if "\n\n" in tmp:
                print(tmp, end="", flush=True)
                break
        
        # 计算性能指标
        end_time = time.time()
        total_time = end_time - start_time
        generation_time = end_time - generation_start_time
        
        # 更新总计数据
        total_tokens_generated += tokens_this_round
        total_inference_time += generation_time
        
        # 计算速度
        current_speed = tokens_this_round / generation_time if generation_time > 0 else 0
        average_speed = total_tokens_generated / total_inference_time if total_inference_time > 0 else 0
        
        # 打印性能统计
        print(f"\n" + "─" * 60)
        print(f"本轮生成: {tokens_this_round} tokens | 用时: {generation_time:.2f}s | 速度: {current_speed:.2f} tokens/s")
        print(f"累计统计: {total_tokens_generated} tokens | 平均速度: {average_speed:.2f} tokens/s")
        print_memory_usage("[当前] ")
        print("─" * 60)
        
        # 可选：清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    else:
        print("!!! Error: please say something !!!")