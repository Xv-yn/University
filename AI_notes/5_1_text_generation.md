# Simple Text Generation

Using the model from the previous chapter, we grab the generated logits 
(remember that these logits at this moment are useless because the model 
can't learn yet) and use these logits to predict the next word to be 
generated.

More specifically:
1. We grab the logits
```
Visualization of logits:
        ^      ["every"   [5,9,8,6,2,5],
Tokens  |       "pokemon" [4,5,2,5,9,4],
        v       "likes"   [1,5,6,2,4,9]]
                           <---------->
                             Relation 
                  with each word in vocabulary
```

2. We grab the highest value (at this point we are taking raw values)
```
Visualization of logits:
        ^      ["every"   [5,(9),8,6,2,5],
Tokens  |       "pokemon" [4,5,2,5,(9),4],
        v       "likes"   [1,5,6,2,4,(9)]]
                             |     |
                             v     |    ... etc.
                           "zoo"   v
                                "ghost"
                           <---------->
                             Relation 
                  with each word in vocabulary
```

3. Based on those values, we grab the relavant indexes and use the tokenizer to 
   turn them back into words


```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
```
