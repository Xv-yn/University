# Self Attention with Weights

Its important to note that we are showing the initial step of training a 
model. We are generating the randomized weights, and generating the output 
based on the randomized weights, WITHOUT changing the weights YET.

Unlike the previous method which directly computes relations between tokens, 
this method uses weights in that calculation, which allow the model to learn. 

Previously:

1. Compare input tokens against each other

2. Normalize comparisons via softmax

3. Combine comparisons into context via Weighted Sum

Now:

1. Initialize Learnable Weights

2. The weights are used to transform the token embeddings into query, key, 
   values

3. Compare queries to keys via dot product

4. Normalize comparisons via softmax

5. Compute weighted sum of values using the softmax weights

```python
class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        # NOTE: there are some issues in the next 2 lines
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        # Everything is find from here onwards
        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))
```

As a result, when calculating WEIGHTS (NOT context vectors), we get this 
kind of output:
```
Tensor(
    [[0.3, 0.4, 0.1, ... ], # Token 1 Relation with other tokens
     [0.4, 0.8, 0.5, ... ], # Token 2 Relation with other tokens
     [0.1, 0.7, 0.3, ... ], # Token 3 Relation with other tokens
     [0.9, 0.8, 0.1, ... ], # Token 4 Relation with other tokens
     [0.4, 0.2, 0.7, ... ]] # Token 5 Relation with other tokens
)
```

Expanded Example:
```
Tensor(
    [["Relationship with Token 1": 0.3, "Relationship with Token 2": 0.4, "Relationship with Token 3": 0.1, ... ], # Token 1 Relation with other tokens
     ["Relationship with Token 1": 0.4, "Relationship with Token 2": 0.8, "Relationship with Token 3": 0.5, ... ], # Token 2 Relation with other tokens
     ["Relationship with Token 1": 0.1, "Relationship with Token 2": 0.7, "Relationship with Token 3": 0.3, ... ], # Token 3 Relation with other tokens
     ["Relationship with Token 1": 0.9, "Relationship with Token 2": 0.8, "Relationship with Token 3": 0.1, ... ], # Token 4 Relation with other tokens
     ["Relationship with Token 1": 0.4, "Relationship with Token 2": 0.2, "Relationship with Token 3": 0.7, ... ]] # Token 5 Relation with other tokens
)
```

## Triu and Dropout

Now we can see how each token can communicate/depend on/relate with each 
other. However, with the current relationship tensor, the model can "peek" 
at the answers.

Similar to how you can answer questions with the answer sheet in front of 
you. Thus, we need to make it such that the model cannot "peek" at the 
answers.

When the model generates text, it should create text based on the previous 
data. Hence, we need to apply the following:

```python
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)
```

This Triu, allows for this output, which removes the ability to see the 
answers:

```
print(masked):
tensor([[0.2899,   -inf,   -inf,   -inf,   -inf,   -inf],
        [0.4656, 0.1723,   -inf,   -inf,   -inf,   -inf],
        [0.4594, 0.1703, 0.1731,   -inf,   -inf,   -inf],
        [0.2642, 0.1024, 0.1036, 0.0186,   -inf,   -inf],
        [0.2183, 0.0874, 0.0882, 0.0177, 0.0786,   -inf],
        [0.3408, 0.1270, 0.1290, 0.0198, 0.1290, 0.0078]],
       grad_fn=<MaskedFillBackward0>)

print(attn_weights):
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5517, 0.4483, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3800, 0.3097, 0.3103, 0.0000, 0.0000, 0.0000],
        [0.2758, 0.2460, 0.2462, 0.2319, 0.0000, 0.0000],
        [0.2175, 0.1983, 0.1984, 0.1888, 0.1971, 0.0000],
        [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
       grad_fn=<SoftmaxBackward0>)
```

Now dropout, which randomly masking out a specified amount of the attention 
weights (usually 10% or 20%), allow for us to prevent overfitting.

```python
self.dropout = nn.Dropout(dropout)

attn_weights = self.dropout(attn_weights)
```

Resulting in the following code:
```python
class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout) # New
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) # New

    def forward(self, x):
        b, num_tokens, d_in = x.shape # New batch dimension b
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2) # Changed transpose
        attn_scores.masked_fill_(  # New, _ ops are in-place
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)  # `:num_tokens` to account for cases where the number of tokens in the batch is smaller than the supported context_size
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights) # New

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)

context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)

context_vecs = ca(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
```
