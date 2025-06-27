# LLM 常用算子

## scale_dynamic_quant
Given hidden_states (**[num_tokens, hidden_size], bfloat16**), mul smooth_scale (**[hidden_size]**) and dynamic quant on hidden_size dim.

- (in) hidden_states
    - [num_tokens, hidden_size]
    - bfloat16
- (in) smooth_scale
    - [hidden_size]
    - float32
- (out) y
    - [num_tokens, hidden_size]
    - int8
- (out) scale
    - [num_tokens]
    - float32



## rms_norm
Given hidden_states (**[num_tokens, hidden_size], bfloat16**), add residual first optionally and rms_norm.

- (in) hidden_states
    - [num_tokens, hidden_size]
    - bfloat16
- (in) residual
    - optional
    - [num_tokens, hidden_size]
    - bfloat16
- (in) weight
    - [hidden_size]
    - float32
- (out) after_res
    - [num_tokens, hidden_size]
    - bfloat16
- (out) y
    - [num_tokens, hidden_size]
    - bfloat16
- (attr) eps
    - float32


## add_rms_norm_dynamic_quant
Given hidden_states (**[num_tokens, hidden_size], bfloat16**), add residual optionally, rms_norm, and then scale_dynamic_quant.

- (in) hidden_states
    - [num_tokens, hidden_size]
    - bfloat16
- (in) residual
    - optional
    - [num_tokens, hidden_size]
    - bfloat16
- (in) weight
    - [hidden_size]
    - bfloat16 or float32
- (in) smooth_scale
    - [hidden_size]
    - bfloat16 or float32
- (out) after_res
    - [num_tokens, hidden_size]
    - bfloat16
- (out) y
    - [num_tokens, hidden_size]
    - int8
- (out) per_token_scale
    - [num_tokens]
    - float32
- (attr) eps
    - float32


## quant_matmul
Given hidden_states (**[num_tokens, hidden_size], int8**), weight (**[hidden_size, new_hidden_size], int8**), matmul and dequant.

- (in) hidden_states:
    - [num_tokens, hidden_size]
    - int8
- (in) per_token_scale
    - required
    - [num_tokens]
    - float32
- (in) weight:
    - [hidden_size, new_hidden_size]
    - int8
- (in) weight_scale:
    - [new_hidden_size]
    - bfloat16 or float32
- (in) bias:
    - optional
    - [new_hidden_size]
    - bfloat16
- (out) y:
    - [num_tokens, new_hidden_size]
    - bfloat16
- (attr) transpose_a
    - bool
- (attr) transpose_b
    - bool



## head_rms_norm
Given hidden_states (**[num_tokens, head_num, head_dim]**, bfloat16), for each token, rms_norm on head_dim dim on specified heads [head_start, head_start + head_num)

- (in) hidden_states:
    - [num_tokens, head_num, head_dim]
    - bfloat16
    - support stride on **head_num**, which means only some heads will be rms_normed.
- (in) weight:
    - [head_num, head_dim]
    - bfloat16 or float32
---
- (out) y:
    - [num_tokens, (q_head_num + 2 * kv_head_num), head_dim]
    - bfloat16
---
- (attr) eps:
    - float32

- (attr) head_offset:
    - int32
- (attr) head_num:
    - int32



## rotary_embedding
1. Given qkv:
unpacked: **[batch_size, q_seq_len, q_head_num + 2 * kv_head_num, head_dim]**  
packed:**[num_tokens, q_head_num + 2 * kv_head_num, head_dim]**, 

2. Pre compute sin and cos

3. Rope on head_dim with **rope_offset** and **rope_dim**
--- 

- (in) qkv
    - [batch_size, q_seq_len, q_head_num + 2 * kv_head_num, head_dim]
      or [num_tokens, q_head_num + 2 * kv_head_num, head_dim]
    - bfloat16
    - only rope on q and k, and split head_dim to nope and rope

- (in) sin
    - [max_position_embedding, rope_dim]
    - bfloat16

- (in) cos
    - [max_position_embedding, rope_dim]
    - bfloat16

- (in) position_ids
    - [batch_size]
    - int32
    - start position id for each seq

- (in) q_lens
    - [batch_size]
    - int32
    - q seq len for each seq

- (in) accum_q_len
    - optional
    - [batch_size + 1]
    - int32
    - accumulated q seq len, exclude current seq
---
- (out) out
    - [batch_size, q_seq_len, q_head_num + 2 * kv_head_num, head_dim]
      or [num_tokens, q_head_num + 2 * kv_head_num, head_dim]
    - bfloat16
---
- (attr) max_q_len
    - int32







## store_kv_cache
1. Given key and value:  
unpacked: **[batch_size, q_seq_len, kv_head_num, head_dim]**  
packed:**[num_tokens, kv_head_num, head_dim]**, 

2. store to k_cache and v_cache:  
**[max_batch_size, kv_head_num, max_seq_len, head_dim]**

3. optionally quantize to int8.
---
- (in) key
    - [batch_size, q_seq_len, kv_head_num, head_dim]  
      or [num_tokens, q_seq_len, kv_head_num, head_dim] 
    - bfloat16
    - kv_head_num may be indices from [q_head_num + 2 * kv_head_num]
- (in) value
    - [batch_size, kv_head_num, head_dim]  
      or [num_tokens, kv_head_num, head_dim]
    - bfloat16
    - kv_head_num may be indices from [q_head_num + 2 * kv_head_num]
- (in) k_cache
    - [max_batch_size, kv_head_num, max_seq_len, head_dim]
    - bfloat16 or int8
- (in) v_cache
    - [max_batch_size, kv_head_num, max_seq_len, head_dim]
    - bfloat16 or int8
- (in) kv_ids
    - [batch_size]
    - int32
    - default is [0, 1, 2, ..., batch_size - 1]
    - cache slot id for each seq
- (in) q_lens
    - [batch_size]
    - int32
- (in) kv_lens
    - optional
    - [batch_size]
    - int32
    - default is [0, 0, 0, ..., 0]
    - past kv len for each seq

- (in) accum_q_len
    - optional
    - [batch_size + 1]
    - int32
    - required for **packed mode**, accumulated query seq len for each seq, exclude current seq

- (in) key_scale
    - optional
    - [kv_head_num, head_dim]
    - float32
- (in) value_scale
    - optional
    - [kv_head_num, head_dim]
    - float32
---
- (out) k_cache
    - [max_batch_size, kv_head_num, max_seq_len, head_dim]
    - bfloat16 or int8
    - input k_cache itself with appended k_cache
- (out) v_cache
    - [max_batch_size, kv_head_num, max_seq_len, head_dim]
    - bfloat16 or int8
    - input v_cache itself with appened v_cache
---
- (attr) max_q_len
    - int32

## store_paged_kv_cache
1. Given key and value:  
unpacked: **[batch_size, q_seq_len, kv_head_num, head_dim]**  
packed:**[num_tokens, kv_head_num, head_dim]**, 

2. store to paged k_cache and v_cache:  
   **[max_block_num, kv_head_num, block_size, head_dim]**  

   with **block_table** specifying block_ids for each block of each seq.  
   **[max_batch_size, max_block_num_per_seq]**

3. optionally quantize to int8.
---
- (in) key
    - [batch_size, q_seq_len, kv_head_num, head_dim]  
      or [num_tokens, q_seq_len, kv_head_num, head_dim] 
    - bfloat16
    - kv_head_num may be indices from [q_head_num + 2 * kv_head_num]
- (in) value
    - [batch_size, kv_head_num, head_dim]  
      or [num_tokens, kv_head_num, head_dim]
    - bfloat16
    - kv_head_num may be indices from [q_head_num + 2 * kv_head_num]
- (in) block_table
    - [max_batch_size, max_block_num_per_seq]
    - int32
    - logical kv_cache
- (in) k_cache
    - [max_block_num, kv_head_num, block_size, head_dim]
    - bfloat16 or int8
- (in) v_cache
    - [max_block_num, kv_head_num, block_size, head_dim]
    - bfloat16 or int8
- (in) kv_ids
    - [batch_size]
    - int32
    - default is [0, 1, 2, ..., batch_size - 1]
    - cache slot id for each seq
- (in) kv_lens
    - optional
    - [batch_size]
    - int32
    - default is [0, 0, 0, ..., 0]
    - past kv len for each seq
- (in) q_lens:
    - [batch_size]
    - int32
    - required for **packed mode**, q len for each seq
- (in) accum_q_len
    - optional
    - [batch_size + 1]
    - int32
    - required for **packed mode**, accumulated query seq len for each seq, exclude current seq

- (in) key_scale
    - optional
    - [kv_head_num, head_dim]
    - float32
- (in) value_scale
    - optional
    - [kv_head_num, head_dim]
    - float32

- (out) k_cache
    - [max_batch_size, kv_head_num, max_seq_len, head_dim]
    - bfloat16 or int8
    - input k_cache itself with appended k_cache
- (out) v_cache
    - [max_batch_size, kv_head_num, max_seq_len, head_dim]
    - bfloat16 or int8
    - input v_cache itself with appened v_cache

- (attr) max_q_len
    - int32


## flash_attention (for benchmark)
Naive flash_attention implementation, used for performance benchmarking, assuming that batch_size = 1, kv_len = 0, q_len > 0.

- (in) qkv
    - required, mutually exclusive with q/k/v.
    - [1, q_seq_len, q_head_num + 2 * kv_head_num, head_dim]  
    - bfloat16

- (in) q
    - required, mutually exclusive with qkv.
    - [1, q_seq_len, q_head_num, head_dim]
    - bfloat16

- (in) k
    - required, mutually exclusive with qkv.
    - [1, q_seq_len, kv_head_num, head_dim]
    - bfloat16

- (in) v
    - required, mutually exclusive with qkv.
    - [1, q_seq_len, kv_head_num, head_dim]
    - bfloat16
---
- (out) out
    - [1, q_seq_len, q_head_num, head_dim]
    - bfloat16
---
- (attr) is_causal
    - bool
    - default is true


## flash_decoding (for benchmark)
Naive flash_decoding implementation, used for performance benchmarking, assuming that batch_size >= 1, kv_len = 0 > 0, q_len = 1, and all kv_lens are equal.

- (in) qkv
    - required, mutually exclusive with q/k/v.
    - [batch_size, 1, q_head_num + 2 * kv_head_num, head_dim]
    - bfloat16

- (in) q
    - required, mutually exclusive with qkv.
    - [batch_size, 1, q_head_num, head_dim]
    - bfloat16

- (in) k
    - required, mutually exclusive with qkv.
    - [batch_size, 1, q_head_num, head_dim]

- (in) k_cache
    - required
    - [max_batch_size, kv_head_num, max_seq_len, head_dim]
    - bfloat16

- (in) v_cache
    - required
    - [max_batch_size, kv_head_num, max_seq_len, head_dim]
    - bfloat16
--- 
- (out) out
    - [batch_size, 1, q_head_num, head_dim]
    - bfloat16
---
- (attr) kv_lens
    - int
    - equal kv_len for all seqs







## flash_attention_prefill
We require:
1. Dedicated for prefill phase

2. Support **remove padding**. If multi-batched qkv given, concat to batch_size = 1 with auxiliary **q_lens** and **kv_lens**.

3. Support **chunked prefill**. **kv_lens** may not be 0 for each seq, used for session cache or chunked pp.

4. Support **paged cache**. chunked prefill needs reading cache.

5. Support **bfloat16 kv_cache and quantized int8 kv_cache**.

---

- (in) q:
    - [num_tokens, q_head_num, head_dim]
    - bfloat16
- (in) kv_ids:
    - [batch_size]
    - int32
- (in) kv_lens:
    - [batch_size]
    - int32
- (in) q_lens:
    - [batch_size]
    - int32
- (in) accum_q_len
    - optional
    - [batch_size + 1]
    - int32
---
- (in) k_cache:
    - [max_block_num, kv_head_num, block_size, head_dim]
    - bfloat16 or int8
- (in) v_cache:
    - [max_block_num, kv_head_num, block_size, head_dim]
    - bfloat16 or int8
- (in) k_scale:
    - optional
    - [kv_head_num, head_dim]
    - float32
- (in) v_scale:
    - optional
    - [kv_head_num, head_dim]
    - float32
---
- (out) out:
    - [num_tokens, q_head_num, head_dim]
    - bfloat16
---
- (attr) max_q_len:
    - int32



## flash_attention_decode
We require:
1. Dedicated for decode phase
2. Support **paged cache**.
3. Support **bfloat16 kv_cache and quantized int8 kv_cache**.
4. kv_len for each seq may be different.
5. q_len for each seq are equal, mostly 1, but 2 or 3 is also possible.
---
- (in) q
    - [batch_size, q_seq_len, head_num, head_dim]
    - bfloat16
- (in) kv_ids:
    - [batch_size]
    - int32
- (in) kv_lens:
    - [batch_size]
    - int32
- (in) q_lens:
    - [batch_size]
    - int32
- (in) accum_q_len:
    - [batch_size + 1]
    - int32
---
- (in) k_cache:
    - [max_block_num, kv_head_num, block_size, head_dim]
    - bfloat16 or int8
- (in) v_cache:
    - [max_block_num, kv_head_num, block_size, head_dim]
    - bfloat16 or int8
- (in) k_scale:
    - optional
    - [kv_head_num, head_dim]
    - float32
- (in) v_scale:
    - optional
    - [kv_head_num, head_dim]
    - float32
---
- (out) out:
    - [batch_size, q_seq_len, head_num, head_dim]
    - bfloat16
---
- (attr) q_seq_len:
    - int32
    - specified q_len for current decode.




## all_reduce
Reduce on hidden_states:  
[num_tokens, hidden_size]



## moe_gating_gemm
Gemm kernel specialized for gating.

- (in) hidden_states
    - [num_tokens, hidden_size]
    - bfloat16

- (in) weight
    - [hidden_size, num_experts]
    - bfloat16

- (out) y
    - [num_tokens, num_experts]
    - bfloat16


## moe_softmax_topk
Given logits after gating_gemm, select topk experts for each token, expert_weights need to be normalized.

- (in) hidden_states
    - required
    - [num_tokens, num_experts]
    - bfloat16
---
- (out) selected_experts
    - [num_tokens, topk]
    - int32

- (out) expert_weights
    - [num_tokens, topk]
    - bfloat16






## moe_scatter_dynamic_quant
In ep (experts-parallel) scenario, each rank will pre-allocated num_experts // world_size weights (up and down). When given selected_experts, scatter input hidden_states to corresponding tensors, dynamic_quant for each token is required either.

About shared experts, we first split on shared experts, and then split on num_tokens if necessary.

---
- (in) hidden_states
    - [num_tokens, hidden_size]
    - bfloat16
- (in) selected_experts
    - [num_tokens, topk]
    - int32
- (in) smooth_scale
    - [num_shared_experts_per_rank + num_experts_per_rank, hidden_size]
    - float32

---
- (out) scatter_tokens
    - [num_shared_experts_per_rank * num_tokens_per_rank + allocated_tokens, hidden_size]
    - bfloat16
    - pre-allocated tensor for current rank

- (out) scatter_per_token_scale
    - [num_shared_experts_per_rank * num_tokens_per_rank + allocated_tokens]
    - float32

- (out) experts_token_count
    - [num_experts_per_rank]
    - int32
    - token count for each expert, **exclude shared_experts**

- (out) token_offset
    - [allocated_tokens, topk]
    - int32
    - output offset for each token, **exclude shared_experts**
---
- (attr) expert_start_idx
    - int
    - for ep, inclusive

- (attr) expert_end_idx
    - int
    - for ep, exclusive

- (attr) num_shared_experts_per_rank
    - int
    - default is 0, if shared_expert is available, append to num_experts_per_rank.

- (attr) num_tokens_offset
    - int
    - if shared_experts need split on num_tokens, this is the token_offset for current rank.

- (attr) num_tokens_per_rank
    - int
    - if shared_experts need split on num_tokens, this is the num_tokens for current rank.




## moe_quant_matmul
Used for shared experts.
- (in) hidden_states
    - [num_tokens_per_rank, hidden_size]
    - int8
- (in) per_token_scale
    - [num_tokens_per_rank]
    - float32
- (in) expert_weight
    - [hidden_size, new_hidden_size]
    - int8
- (in) expert_scale
    - [new_hidden_size]
    - float32
---
- (out) y
    - [num_tokens_per_rank, new_hidden_size]
    - bfloat16


## moe_quant_group_gemm
For each expert, tokens matmul with respective weight.
- (in) scatter_tokens
    - [allocated_tokens, hidden_size]
    - int8
- (in) per_token_scale
    - [allocated_tokens]
    - float32
- (in) experts_weight
    - [num_experts_per_rank, hidden_size, new_hidden_size]
    - int8
- (in) experts_scale
    - [num_experts_per_rank, new_hidden_size]
    - float32
- (in) experts_token_count
    - [num_experts_per_rank]
    - int32
    - token count for each expert.
---
- (out) y
    - [allocated_tokens, new_hidden_size]
    - bfloat16



## moe_swiglu_dynamic_quant
For each expert, tokens calculate swiglu and dynamic quant.
- (in) scatter_tokens
    - [num_shared_experts_per_rank * num_tokens_per_rank + allocated_tokens, hidden_size * 2]
    - bfloat16
- (in) smooth_scale
    - [num_shared_experts_per_rank + num_experts_per_rank, hidden_size]
    - float32
    - per expert, need matching scatter_tokens
- (in) experts_tokens_count
    - [num_experts_per_rank]
    - per expert, used to match scatter_tokens
---
- (out) y
    - [num_shared_experts_per_rank * num_tokens_per_rank + allocated_tokens, hidden_size]
    - int8
- (out) per_token_scale
    - [num_shared_experts_per_rank * num_tokens_per_rank + allocated_tokens]
    - float32
---
- (attr) num_shared_experts_per_rank
    - int
    - default is 0, if shared_expert is available, append to num_experts_per_rank.

- (attr) num_tokens_offset
    - int
    - if shared_experts need split on num_tokens, this is the token_offset for current rank.

- (attr) num_tokens_per_rank
    - int
    - if shared_experts need split on num_tokens, this is the num_tokens for current rank.



## moe_gather
At experts-parallel scenario, each rank will calculate respective tokens, and need gathering back and index_add. 
$$dst_j = dst_j + src_i * weight_k$$

j/k is determined by token_offset, and selected from scatter_tokens/expert_weights.
In addition, shared experts for current rank will also be accumulated to output.


- (in) scatter_tokens
    - [num_shared_experts_per_rank * num_tokens_per_rank + allocated_tokens, hidden_size]
    - bfloat16
- (in) token_offset
    - [allocated_tokens]
    - int32
- (in) expert_weights
    - [num_tokens, topk]
    - float32
---
- (out) hidden_states
    - [num_tokens, hidden_size]
    - bfloat16
    - pre-allocated, and initialized to zero
---
- (attr) num_shared_experts_per_rank
    - int
    - default is 0, if shared_expert is available, append to num_experts_per_rank.

- (attr) num_tokens_offset
    - int
    - if shared_experts need split on num_tokens, this is the token_offset for current rank.

- (attr) num_tokens_per_rank
    - int
    - if shared_experts need split on num_tokens, this is the num_tokens for current rank.

