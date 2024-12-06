from collections import namedtuple
import json
import pandas as pd 
import itertools
from openpyxl import Workbook
 

TFLOPS = 1e12
GB = 1024*1024*1024

vocab_size = 32768
hidden_size = 6144
intermediate_size = 16384
num_heads = 48
num_kv_heads = 8
num_kv_groups = 6
num_experts_per_tok = 2
num_layers = 56
num_experts = 8
head_dim = hidden_size // num_heads
is_silu = True


res = {"qkv_proj": {},
       "qk_dot": {},
       "pv_dot": {},
       "out_proj": {},
       "attn": {},
       "gate_fc": {},
       "fc1_0": {},
       "fc1_1": {},
       "fc2": {},
       "moe": {},
       "lm_head": {},
    #    "tot_block": {},
       "tot_layer": {}
       }

def load_model_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    global vocab_size
    global hidden_size
    global intermediate_size
    global num_heads
    global num_kv_heads
    global num_kv_groups
    global num_experts_per_tok, num_layers, num_experts, head_dim, is_silu
    vocab_size = config["network"]["vocab_size"]
    hidden_size = config["network"]["hidden_size"]
    intermediate_size = config["network"]["intermediate_size"]
    num_heads = config["network"]["num_attention_heads"]
    num_kv_heads = config["network"]['num_key_value_heads']
    num_kv_groups = num_heads / num_kv_heads
    num_experts_per_tok = config["network"]["num_experts_per_tok"]
    num_layers = config["network"]["num_hidden_layers"]
    num_experts = config["network"]["num_local_experts"]
    head_dim = hidden_size // num_heads

    # TODO: Check if silu in P5 model config!
    # check if is silu
    if config["network"]["hidden_act"] == "silu":
        is_silu = True
    else:
        is_silu = False
    return config

def calc_gemm(M, K, N, bias=False):
    flops = 2 * M * K * N
    if not bias:
        flops -= M * N
    return flops, M * K, K * N, M * N

def calc_batch_gemm(batch_size, M, K, N, bias=False):
    flops = batch_size * 2 * M * K * N
    if not bias:
        flops -= batch_size * M * N
    return flops, batch_size * M * K, batch_size * K * N, batch_size * M * N


def calc_group_gemm(num_group, total_M, K, N, bias=False):
    flops = 2 * total_M * K * N
    if not bias:
        flops -= total_M * N
    return flops, total_M * K, num_group * K * N, total_M * N


def calc_mfu(batch_size, q_seq_len, kv_seq_len, latency_ms, peak_tensor_tflops):
    if q_seq_len == 1 and kv_seq_len != 0:
        is_context = False
    elif q_seq_len != 0 and kv_seq_len == 0:
        is_context = True
    else:
        raise RuntimeError("error")

    qkv_proj, qkv_left, qkv_right, qkv_out = calc_gemm(batch_size * q_seq_len, hidden_size, (num_heads + 2 * num_kv_heads) * head_dim)
    print(f"qkv proj tflops = {qkv_proj / 1e12} TFLOPS")

    res["qkv_proj"]["compute"] = qkv_proj/TFLOPS

    if is_context:
        kv_seq_len = q_seq_len
        qk_dot, qk_left, qk_right, qk_out = calc_batch_gemm(batch_size * num_heads, q_seq_len, head_dim, kv_seq_len)
        
        pv_dot, pv_left, pv_right, pv_out = calc_batch_gemm(batch_size * num_heads, q_seq_len, kv_seq_len, head_dim)
    else:
        kv_seq_len = kv_seq_len + q_seq_len
        qk_dot, qk_left, qk_right, qk_out = calc_batch_gemm(batch_size * num_heads, q_seq_len, head_dim, kv_seq_len)    
        pv_dot, pv_left, pv_right, pv_out = calc_batch_gemm(batch_size * num_heads, q_seq_len, kv_seq_len, head_dim)

    print(f"qk dot tflops = {qk_dot/TFLOPS} TFLOPS")
    print(f"pv dot tflops = {pv_dot/TFLOPS} TFLOPS")
    res["qk_dot"]["compute"] = qk_dot/TFLOPS
    res["pv_dot"]["compute"] = pv_dot/TFLOPS
        
    out_proj, out_left, out_right, out_out = calc_gemm(batch_size * q_seq_len, hidden_size, hidden_size)
    print(f"out proj tflops = {out_proj/TFLOPS} TFLOPS")
    res["out_proj"]["compute"] = out_proj / 1e12

    # NOTE: include qkv_proj
    attn_flops = qkv_proj + qk_dot + pv_dot + out_proj
    print(f"attn flops = {attn_flops/TFLOPS} TFLOPS")
    res["attn"]["compute"] = attn_flops/TFLOPS


    # SwiGLU 是在Mixtral和Seed P6、P7模型中的，需要3个w1、w2、w3
    gate_fc, gate_left, gate_right, gate_out = calc_gemm(batch_size * q_seq_len, hidden_size, num_experts)
    print(f"gate fc tflops = {gate_fc/TFLOPS} TFLOPS")
    res["gate_fc"]["compute"] = gate_fc / 1e12

    w1, w1_left, w1_right, w1_out = calc_group_gemm(num_experts, batch_size * q_seq_len * num_experts_per_tok, hidden_size, intermediate_size)
    print(f"ffn w1 tflops = {w1 / 1e12} TFLOPS")
    res["fc1_0"]["compute"] = w1/TFLOPS

    w2, w2_left, w2_right, w2_out = calc_group_gemm(num_experts, batch_size * q_seq_len * num_experts_per_tok, intermediate_size, hidden_size)
    print(f"ffn w2 tflops = {w2/TFLOPS} TFLOPS")
    res["fc1_1"]["compute"]  = w2/TFLOPS

    w3, w3_left, w3_right, w3_out = calc_group_gemm(num_experts, batch_size * q_seq_len * num_experts_per_tok, hidden_size, intermediate_size)
    print(f"ffn w3 tflops = {w3/TFLOPS} TFLOPS")
    res["fc2"]["compute"] = w3/TFLOPS

    moe_flops = gate_fc + w1 + w2 + w3
    print(f"moe flops = {moe_flops/TFLOPS} TFLOPS")
    res["moe"]["compute"] = moe_flops / 1e12

    lm_head_flops, lm_left, lm_right, lm_out = calc_gemm(batch_size * q_seq_len, hidden_size, vocab_size)
    print(f"lm head flops = {lm_head_flops / 1e12} TFLOPS")
    res["lm_head"]["compute"] = lm_head_flops / 1e12

    # calculate total flops
    # calculate total flops
    total_flops = (num_layers * (attn_flops + moe_flops) + lm_head_flops) / 1e12

    print(f"total flops = {total_flops} TFLOPS")
    res["tot_layer"]["compute"] = total_flops
    test_flops = total_flops / (latency_ms * 1e-3)
    mfu = test_flops / peak_tensor_tflops

    return round(total_flops, 3), round(test_flops, 3), round(mfu, 3)



def calc_mbu(batch_size, q_seq_len, kv_seq_len, latency_ms, peak_mem_bw):
    if q_seq_len == 1 and kv_seq_len != 0:
        is_context = False
    elif q_seq_len != 0 and kv_seq_len == 0:
        is_context = True
    else:
        raise RuntimeError("error")

    qkv_proj, qkv_left, qkv_right, qkv_out = calc_gemm(batch_size * q_seq_len, hidden_size, (num_heads + 2 * num_kv_heads) * head_dim)

    # res["qkv_proj"]["input"] = qkv_left / GB
    res["qkv_proj"]["weight"] = qkv_right / GB
    # res["qkv_proj"]["output"] = qkv_out / GB
    res["qkv_proj"]["inout"] = qkv_left / GB + qkv_out / GB

    print(f"qkv proj input&output = {res['qkv_proj']['inout']} GB, weight = {res['qkv_proj']['weight']} GB")

    if is_context:
        kv_seq_len = q_seq_len
        qk_dot, qk_left, qk_right, qk_out = calc_batch_gemm(batch_size * num_heads, q_seq_len, head_dim, kv_seq_len)
        pv_dot, pv_left, pv_right, pv_out = calc_batch_gemm(batch_size * num_heads, q_seq_len, kv_seq_len, head_dim)
    else:
        kv_seq_len = kv_seq_len + q_seq_len
        qk_dot, qk_left, qk_right, qk_out = calc_batch_gemm(batch_size * num_heads, q_seq_len, head_dim, kv_seq_len)
        pv_dot, pv_left, pv_right, pv_out = calc_batch_gemm(batch_size * num_heads, q_seq_len, kv_seq_len, head_dim)


    res["qk_dot"]["weight"] = qk_right / GB
    res["qk_dot"]["inout"] = (qk_left + qk_out) / GB 

    res["pv_dot"]["weight"] = pv_right / GB
    res["pv_dot"]["inout"] = (pv_left + pv_out) / GB 

    print(f"qk dot input&output = {res['qk_dot']['inout']} GB, weight = {res['qk_dot']['weight']} GB")

    print(f"pv dot input&output = {res['pv_dot']['inout']} GB, weight = {res['pv_dot']['weight']} GB")


    out_proj, out_left, out_right, out_out = calc_gemm(batch_size * q_seq_len, hidden_size, hidden_size)
    
    res["out_proj"]["weight"] = out_right / GB
    res["out_proj"]["inout"] = (out_left + out_out) / GB
    print(f"out proj input&output = {res['out_proj']['inout']} GB, weight = {res['out_proj']['weight']} GB")

    # calculate input and output size == activations, except weights
    attn_inout = 0
    attn_inout += qkv_left + qkv_out

    # on-FlashAttn, need read & write Partial result
    attn_inout += qk_left + qk_right + qk_out
    attn_inout += pv_left + pv_right + pv_out
    attn_inout += out_left + out_out

    # calculate model weights memory size!
    attn_weight = 0
    attn_weight += qkv_right
    attn_weight += out_right

    res["attn"]["inout"] = attn_inout / GB
    res["attn"]["weight"] = attn_weight / GB
    print(f"attn inout = {res['attn']['inout']} GB, weight = {res['attn']['weight']} GB")


    gate_fc, gate_left, gate_right, gate_out = calc_gemm(batch_size * q_seq_len, hidden_size, num_experts)
    w1, w1_left, w1_right, w1_out = calc_group_gemm(num_experts, batch_size * q_seq_len * num_experts_per_tok, hidden_size, intermediate_size)
    w2, w2_left, w2_right, w2_out = calc_group_gemm(num_experts, batch_size * q_seq_len * num_experts_per_tok, intermediate_size, hidden_size)
    w3, w3_left, w3_right, w3_out = calc_group_gemm(num_experts, batch_size * q_seq_len * num_experts_per_tok, hidden_size, intermediate_size)

    res["gate_fc"]["inout"] = (gate_left+gate_out) / GB
    res["gate_fc"]["weight"] = gate_right / GB

    res["fc1_0"]["inout"] = (w1_left+w1_out) / GB
    res["fc1_0"]["weight"] = w1_right / GB

    res["fc1_1"]["inout"] = (w2_left+w2_out) / GB
    res["fc1_1"]["weight"] = w2_right / GB

    res["fc2"]["inout"] = (w3_left+w3_out) / GB
    res["fc2"]["weight"] = w3_right / GB

    print(f"gate fc inout = {res['gate_fc']['inout']} GB, weight = {res['gate_fc']['weight']} GB")
    print(f"fc1_0 inout = {res['fc1_0']['inout']} GB, weight = {res['fc1_0']['weight']} GB")
    print(f"fc1_1 inout = {res['fc1_1']['inout']} GB, weight = {res['fc1_1']['weight']} GB")
    print(f"fc2 inout = {res['fc2']['inout']} GB, weight = {res['fc2']['weight']} GB")

    moe_inout = 0
    moe_inout += gate_left + gate_out
    moe_inout += w1_left + w1_out
    moe_inout += w2_left + w2_out
    moe_inout += w3_left + w3_out

    moe_weight = 0
    moe_weight += gate_right
    moe_weight += w1_right
    moe_weight += w2_right
    moe_weight += w3_right

    res["moe"]["inout"] = moe_inout / GB   
    res["moe"]["weight"] = moe_weight / GB 
    print(f"moe inout = {res['moe']['inout']} GB, weight = {res['moe']['weight']} GB")


    lm_head_flops, lm_left, lm_right, lm_out = calc_gemm(batch_size * q_seq_len, hidden_size, vocab_size)

    lm_head_inout = lm_left + lm_out
    lm_head_weight = lm_right

    res["lm_head"]["inout"] = lm_head_inout / GB
    res["lm_head"]["weight"] = lm_head_weight / GB
    print(f"lm_head inout = {res['lm_head']['inout']} GB, weight = {res['lm_head']['weight']} GB")

    total_inout = (num_layers * (attn_inout + moe_inout) + lm_head_inout)
    total_weight = (num_layers * (attn_weight + moe_weight) + lm_head_weight)
    total_mem_size = (total_inout + total_weight) / 1e9
    res["tot_layer"]["inout"] = total_inout / GB 
    res["tot_layer"]["weight"] = total_weight / GB

    print(f"total inout = {res['tot_layer']['inout']} GB, weight = {res['tot_layer']['weight']} GB")
    res["tot_layer"]["mem_size"] = total_mem_size / GB

    print(f"total mem size = {total_mem_size} GB")
    test_mem_bw = total_mem_size / (latency_ms * 1e-3)
    mbu = test_mem_bw / peak_mem_bw

    return round(total_mem_size, 3), round(test_mem_bw, 3), round(mbu, 3)



# config different model
config_file = "./models/mixtral-torch-bf16-8x22b.json"
load_model_config(config_file)


DeployConfig = namedtuple('DeployConfig', ['tp_size', 'peak_tensor_tflops', 'peak_mem_bw'])
TestConfig = namedtuple('TestConfig', ['batch_size', 'q_seq_len', 'kv_seq_len', 'latency_ms'])


MI308X_deploy_config = DeployConfig(tp_size=8, peak_tensor_tflops=233, peak_mem_bw=4000)

# NOTE: config different batch size & seq len
batch_size_list = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 264, 272, 280, 288, 296, 304, 312, 320, 328, 336, 344, 352, 360, 368, 376, 384, 392, 400, 408, 416, 424, 432, 440, 448, 456, 464, 472, 480, 488, 496, 504, 512]

seq_len_list = [1024, 2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624, 28672, 30720, 32768]
context_latency_list = [265.97, 478.81, 710.16, 926.85, 1126.85, 1326.85, 1526.85, 1726.85, 1926.85, 2126.85, 2326]

TestConfig(batch_size=1, q_seq_len=8192, kv_seq_len=0, latency_ms=265.97)


# prefill config
context_test_list = [
    # TODO：determine each latency
    TestConfig(batch_size=1, q_seq_len=8192, kv_seq_len=0, latency_ms=265.97) 
    # TestConfig(batch_size=1, q_seq_len=2048, kv_seq_len=0, latency_ms=265.97),
    # TestConfig(batch_size=1, q_seq_len=4096, kv_seq_len=0, latency_ms=478.81),
    # TestConfig(batch_size=1, q_seq_len=6144, kv_seq_len=0, latency_ms=710.16),
    # TestConfig(batch_size=1, q_seq_len=8192, kv_seq_len=0, latency_ms=926.85),
    # TestConfig(batch_size=1, q_seq_len=10240, kv_seq_len=0, latency_ms=926.85),
    # TestConfig(batch_size=1, q_seq_len=12288, kv_seq_len=0, latency_ms=926.85),
    # TestConfig(batch_size=1, q_seq_len=14336, kv_seq_len=0, latency_ms=926.85),
    # TestConfig(batch_size=1, q_seq_len=16384, kv_seq_len=0, latency_ms=926.85),
    # TestConfig(batch_size=1, q_seq_len=184, kv_seq_len=0, latency_ms=926.85),
    # TestConfig(batch_size=1, q_seq_len=8192, kv_seq_len=0, latency_ms=926.85),

    # TestConfig(batch_size=1, q_seq_len=16384, kv_seq_len=0, latency_ms=1897.18),

    # for seq_len, latency in zip(seq_len_list, context_latency_list)
]

# decode_test_list = [
#     # TestConfig(batch_size=1, q_seq_len=1, kv_seq_len=8192, latency_ms=11.67),
#     # TestConfig(batch_size=4, q_seq_len=1, kv_seq_len=8192, latency_ms=16.41),
#     # TestConfig(batch_size=8, q_seq_len=1, kv_seq_len=8192, latency_ms=19.51),
#     # TestConfig(batch_size=16, q_seq_len=1, kv_seq_len=8192, latency_ms=20.94),
#     # TestConfig(batch_size=32, q_seq_len=1, kv_seq_len=8192, latency_ms=25.15),
#     # TestConfig(batch_size=40, q_seq_len=1, kv_seq_len=8192, latency_ms=27.45),
#     # TestConfig(batch_size=48, q_seq_len=1, kv_seq_len=8192, latency_ms=29.61),
#     # TestConfig(batch_size=64, q_seq_len=1, kv_seq_len=8192, latency_ms=33.08),
#     # TestConfig(batch_size=80, q_seq_len=1, kv_seq_len=8192, latency_ms=36.24),
#     # TestConfig(batch_size=96, q_seq_len=1, kv_seq_len=8192, latency_ms=41.72),
#     # TestConfig(batch_size=112, q_seq_len=1, kv_seq_len=8192, latency_ms=45.65),
#     # TestConfig(batch_size=120, q_seq_len=1, kv_seq_len=8192, latency_ms=49.47),
#     TestConfig(batch_size=128, q_seq_len=1, kv_seq_len=8192, latency_ms=50.9)
#     for bs in batch_size_list for seq_len in seq_len_list
# ]



# print("prefill")
for test_config in context_test_list:

    tps_per_device = round(1000 / test_config.latency_ms * test_config.batch_size / MI308X_deploy_config.tp_size, 3) * test_config.q_seq_len

    results = calc_mfu(
        batch_size=test_config.batch_size,
        q_seq_len=test_config.q_seq_len,
        kv_seq_len=test_config.kv_seq_len,
        latency_ms=test_config.latency_ms,
        peak_tensor_tflops=MI308X_deploy_config.peak_tensor_tflops * MI308X_deploy_config.tp_size
    )
    model_tflops, test_tflops, mfu = results

    results = calc_mbu(
        batch_size=test_config.batch_size,
        q_seq_len=test_config.q_seq_len,
        kv_seq_len=test_config.kv_seq_len,
        latency_ms=test_config.latency_ms,
        peak_mem_bw=MI308X_deploy_config.peak_mem_bw * MI308X_deploy_config.tp_size
    )
    model_mem_size, test_mem_bw, mbu = results
    print(f"{test_config}: mfu = {mfu}, mbu = {mbu}, tps = {tps_per_device}, model_tflops = {model_tflops} TFLOPS, model_mem_size = {model_mem_size} GB")
print("\n")




# print("decode")
# for test_config in decode_test_list:
#     tps_per_device = round(1000 / test_config.latency_ms * test_config.batch_size / MI308X_deploy_config.tp_size, 3)

#     results = calc_mfu(
#         batch_size=test_config.batch_size,
#         q_seq_len=test_config.q_seq_len,
#         kv_seq_len=test_config.kv_seq_len,
#         latency_ms=test_config.latency_ms,
#         peak_tensor_tflops=MI308X_deploy_config.peak_tensor_tflops * MI308X_deploy_config.tp_size
#     )
#     model_tflops, test_tflops, mfu = results

#     results = calc_mbu(
#         batch_size=test_config.batch_size,
#         q_seq_len=test_config.q_seq_len,
#         kv_seq_len=test_config.kv_seq_len,
#         latency_ms=test_config.latency_ms,
#         peak_mem_bw=MI308X_deploy_config.peak_mem_bw * MI308X_deploy_config.tp_size
#     )
#     model_mem_size, test_mem_bw, mbu = results
#     # print(f"{test_config}: mfu = {mfu}, mbu = {mbu}, tps = {tps_per_device}, model_tflops = {model_tflops} TFLOPS, model_mem_size = {model_mem_size} GB")
# print("\n")



df = pd.DataFrame(res)
df.to_excel("results.xlsx", index=True)
print(f"Result saved to results.xlsx")




