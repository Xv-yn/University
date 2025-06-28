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
        
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))
```
