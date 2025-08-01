# Combining Everything Into a Simple Model

What we are doing here is running the model (not yet able to "learn") and 
getting it to generate text based on the randomly generated weights.

The general process is as follows:
1. Initialization
    - Random token embeddings
    - Random positional embeddings
    - Dropout Rate
    - Transformer Blocks (x12)
    - Randomly weighted normalization layer
    - `out_head`

2. We add the token embeddings to the positional embeddings giving us 
   matrix `x`
    - At this point, `x` is technically context vectors but has no context
    - `x` has dimensions `[batch, tokens (words in a sentence), embeddings]`

3. We apply dropout to `x`
    - `x` has dimensions `[batch, tokens (words in a sentence), embeddings]`

4. We put `x` through the transformer blocks
    - Inside the transformer blocks, `x` now has at least some sort of 
      context based on token positions and can be properly called context 
      vectors
    - `x` has dimensions `[batch, tokens (words in a sentence), embeddings]`

5. We normalize all values in `x`
    - `x` has dimensions `[batch, tokens (words in a sentence), embeddings]`

6. We multiply `x` by `head_out` which gives us `logits`
    - `logits` are the raw values showing how each word in the vocabulary 
      relates to each token
    - `logits` have dimensions `[token (words in a sentence), vocab]` 

```python
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
```

