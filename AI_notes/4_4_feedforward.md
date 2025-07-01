# Feedforward

We will NOT complete the feedforward network here, but will provide a 
structure that will be used to finish it.

Here is the general idea:
- Given the embedding dimensions, when the model is learning in each 
  iteration, we will increase the "thinking space" by a factor of 4. We will 
  add the "thinking and learning" part later.

Here is the structure in python:
```python
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)
```
