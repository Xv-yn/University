# Transformer Block

The transformer block runs as follows:

1. We initalize the block based on the initial configuration
    - Multihead Attention:
        - attention_score = key @ query
        - triu mask
        - softmax
        - dropout
        - context_vectors = attention_score @ values
    - Feed Forward
        - Linear (increase dimensions)
        - GELU
        - Linear (revert dimensions)
    - Normalization Layers
        - Create Normalization Weights and Biases
        - Prepare Normalization Function
    - Dropout
        - Prepare dropout function
2. In a forwards pass:
    - Normalize Layer 1
    - Multihead Attenion
    - Dropout Function
    - Add Original Input for Shortcut
    - Normalize Layer 2
    - Feed Forward
    - Dropout Function
    - Add Shortcut for Shortcut

```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x
```

As of right now, what stores the "knowledge", or what can be changed in 
training (WHICH IS NOT DONE YET) is as follows:
- Weights in Token Embeddings
    - These represent the meaning of tokens and are updated via training
- Weights and Biases in the Feed Forward Network
    - Located in the `nn.Linear()`
    - These transform and refine the hidden representations
- Weights and Biases in the Layer Normalization Layers
    - Located in the `LayerNorm()`
    - These are learnable and updated during training, but stay fixed during 
      each forward pass
