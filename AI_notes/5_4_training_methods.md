# Temperature and Top-K Sampling

A common word that you'll hear when talking about LLMs is "temperature".

Temperature is a short hand way of saying "Temprature Scaling", which 
basically means dividing logits (see 4_7) by a values greater than 0.

By doing this we can achieve 1 of 2 results:
- Dividing by numbers > 1 will result in more uniformly distributed token 
  probabilities after softmax
- Dividing by numbers > 0 and < 1 will result in more confident (sharper/
  peaky) distributions after softmax

Here is a live example:
```
Inputs:
logits = [2.0,     1.0,   0.1,    3.0,    0.5]
words  = ["cat", "dog", "hat", "tree", "book"]
```

Temperature = 1.0 (Normal)
```
softmax([2.0, 1.0, 0.1, 3.0, 0.5]) ≈ [0.20, 0.07, 0.03, 0.66, 0.05]
                                                         ^
                                                       "tree" - normally
```

Temperature = 0.5 (Low)
```
logits_scaled = [x / 0.5 for x in logits] = [4.0, 2.0, 0.2, 6.0, 1.0]

softmax(...) ≈ [0.12, 0.01, 0.00, 0.85, 0.01]
                                   ^
                                 "tree" - much bigger than other values
```

Temperature = 2.0 (High)
```
logits_scaled = [x / 2.0 for x in logits] = [1.0, 0.5, 0.05, 1.5, 0.25]
softmax(...) ≈ [0.24, 0.15, 0.10, 0.33, 0.17]
                                   ^
                                 "tree" - somewhat bigger than other values

```

Top-K Sampling is similar to a beam search, where we select the most likely 
K words that are probable to appear.

However, instead of expansion, using those probabilities, we compute another 
softmax and "randomly" choose a word based on the calculated probabilities.

Here is an example:
```
Probabilities after softmax (Temperature=1): 
[cat: 0.20, dog: 0.07, hat: 0.03, tree: 0.66, book: 0.05]

Top-K = 3 → Keep: "tree", "cat", "dog"
Normalize these:
["tree": 0.66, "cat": 0.20, "dog": 0.07] → Renormalized to sum to 1

Then:
Sample randomly *within those 3* based on their (re-normalized) 
probabilities.
```

Here is the integration of this into the word generation function:
```python
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx
```

