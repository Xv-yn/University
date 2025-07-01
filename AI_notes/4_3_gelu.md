# ReLU and GELU

ReLU stands for Rectified Linear Unit.

GELU stands for Gaussian Error Linear Unit.

ReLU is basically:
- Given a list of numbers
    - Iterate through using the follwoing conditions:
        - If number < 0: Set number = 0
        - Else: Number stays the same


GELU is computed differently:
- There are many ways to do this, but the most common is the cheaper option
```python
gelu = 0.5 * x * (1 + 
            torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
```

We use GELU over ReLU because GELU produces a smooth curved graph, allowing 
for a smoother gradient (which doesn't mean much to us yet).

In simpler terms, GELU causes negative values to be partially suppressed but 
not harshly zeroed out, letting us understand more information because 
sometimes, small negative inputs carry useful information — GELU preserves 
that possibility instead of throwing it away completely like ReLU does.



