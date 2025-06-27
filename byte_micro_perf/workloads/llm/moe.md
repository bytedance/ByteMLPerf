# moe logic

### Glossary
- batch_size
- q_seq_len
- hidden_size
- num_experts
- num_topk
- num_tokens: batch_size * q_seq_len
- total_num_tokens: num_tokens * num_topk
- allocated_num_tokens
- average_num_tokens_per_expert: total_num_tokens / num_experts



### hidden_states
- shape: [batch_size, q_seq_len, hidden_size]
- dtype: torch.bfloat16

reshape to [num_tokens, hidden_size]



### gating gemm
```python
router_logits = gating_gemm(hidden_states)
```

- left: [num_tokens, hidden_size]
- right: [hidden_size, num_experts]
- output: [num_tokens, num_experts]


### softmax
```python
routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
```

- input: [num_tokens, num_experts], float32
- output: [num_tokens, num_experts], float32



### topk and normalize
```python
routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
routing_weights = routing_weights.to(hidden_states.dtype)
```

- routing_weights: [num_tokens, num_topk], bfloat16
- selected_experts: [num_tokens, num_topk], int64


### histogram, index, scatter
For each token, we can get **num_topk** expert indexs, which correspond to **selected_experts**.

There will be **total_num_tokens** tokens in total, we will create corresponding tensor **prepared_tokens**.

For each expert, we count how many tokens are routed to it, and then we can get offsets in **expert_token_count** for each token.

Finally, scatter original **hidden_states** to **prepared_tokens** according to **token_offsets**.


| tensor | shape | dtype |
|-|-|-|
| prepared_tokens       |   [total_num_tokens, hidden_size] |   bfloat16 |
| expert_token_count    |   [num_experts]                   |   int64    |
| token_offsets         |   [num_tokens, num_topk]          |   int64    |


### group_gemm
Using group_gemm kernel, and each problem will have different **allocated_num_tokens**.

To simplify performance calculations, we assume that tokens are evenly distributed across different experts, and each expert will have **average_num_tokens_per_expert** tokens.

| tensor | shape | dtype |
|-|-|-|
| prepared_tokens       |   [num_experts, average_num_tokens_per_expert, hidden_size]   |   bfloat16 |
| experts_weight        |   [num_experts, hidden_size, ffn_size]                        |   bfloat16    |
|                       |   [num_experts, ffn_size, hidden_size]                        |   bfloat16    |
| output_tokens         |   [num_experts, average_num_tokens_per_expert, hidden_size]   |   bfloat16    |
|                       |   [total_num_tokens, hidden_size]                             |   bfloat16    |


### gather
Based on **token_offsets** and **output_tokens**, we can gather the final result, during which for each output token we also need to use **num_topk** token tensors (from **output_tokens**) and **num_topk** weight tensors (from **routing_weights**) to calculate the final result.


| tensor | shape | dtype |
|-|-|-|
| output_tokens         |   [total_num_tokens, hidden_size]     |   bfloat16 |
| token_offsets         |   [num_tokens, num_topk]              |   int64    |
| routing_weights       |   [num_tokens, num_topk]              |   bfloat16 |
| final_results         |   [num_tokens, hidden_size]           |   int64    |
















