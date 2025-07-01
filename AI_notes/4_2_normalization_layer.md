# Layer Normalization

Layer normalization is basically grabbing data at a certain point in the 
model learning process and normalizes it with a mean of 0 and variance of 1.

It is applied both before and after multihead attention and also before the 
final output.

```python
torch.manual_seed(123)

# create 2 training examples with 5 dimensions (features) each
batch_example = torch.randn(2, 5) 

layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)

# calculates unnormalized mean and variance
mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)

print("Mean:\n", mean)
print("Variance:\n", var)
```

In the above example, we create 1 batch that has 2 words and 5 dimensions.

We then calculate out the mean.

Afterwards, we normalize the data by subtracting mean and dividing by square 
root of the variance (standard deviation).

```python
out_norm = (out - mean) / torch.sqrt(var)
print("Normalized layer outputs:\n", out_norm)

mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)
```

> [!note]
> Note that the mean will not be exactly 0, it will be very close though

Putting it into the model would look like this:
```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
```

For the most part, this is the same, but theres the addition of scale and 
shift.

What this is is just another way the AI can learn. It's basically weights = 
scale and bias = shift. Like a perceptron.

