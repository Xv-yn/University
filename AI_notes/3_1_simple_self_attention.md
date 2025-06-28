# Simple Self Attention

This is the inner workings of a singular CLUSTER of neurons. Similar to a 
a cluster of perceptrons from COMP3308.

And each row of a weight matrix behaves like one neuron.

Given a bunch of input vectors (`input_vector = token_embedding` in this case) 
we want to calculate the context vector.

To do this, we will use an example to visualize it step by step:
```python
inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your     (x^1)
        [0.55, 0.87, 0.66],  # journey  (x^2)
        [0.57, 0.85, 0.64],  # starts   (x^3)
        [0.22, 0.58, 0.33],  # with     (x^4)
        [0.77, 0.25, 0.10],  # one      (x^5)
        [0.05, 0.80, 0.55],  # step     (x^6)
    ] 
)
```

Given the above inputs, let's use the second input token as the example. We 
calculate the dot product of the second input token and each of the inputs 
to produce attention scores.

> [!note]
> Attention Scores are a direct relationship between two tokens, but with no 
> reference, meaning that in order to understand this relationship, the 
> attention score must be normalized

```python
query = inputs[1]  # 2nd input token is the query

attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query) # dot product (transpose not necessary here since they are 1-dim vectors)

print(attn_scores_2)
```

To understand the relationship of two token relative to other tokens, we 
normalize the values to get attention weights:

```python
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)

print("Attention weights:", attn_weights_2_naive)
```

Hence, to get the final context vectors, we do a weighted sum against the 
second input token. 

```python
query = inputs[1] # 2nd input token is the query

context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i

print(context_vec_2)
```

Thus, this gives us the context of the 2nd input token relative to all the 
other tokens.

This can be summarized into the following code:
```python
inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your     (x^1)
        [0.55, 0.87, 0.66],  # journey  (x^2)
        [0.57, 0.85, 0.64],  # starts   (x^3)
        [0.22, 0.58, 0.33],  # with     (x^4)
        [0.77, 0.25, 0.10],  # one      (x^5)
        [0.05, 0.80, 0.55],  # step     (x^6)
    ] 
)

# dot product comparing each input token against each other via 
# matrix multiplication
attention_scores = inputs @ inputs.T

# normalization via softmax
attn_weights = torch.softmax(attention_scores, dim=-1)

# weighted sum via matrix multiplication
context_vectors = attn_weights @ inputs

print(context_vectors)
```

