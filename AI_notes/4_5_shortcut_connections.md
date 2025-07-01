# Shortcut Connections

Optimization is everything. When the model learns, using this shortcut method 
allows for an alternative shorter path for the model to learn.

Simply put, its similar to how we find it easier to learn through analogies 
rather than through forced memorization.

The way shortcuts work is as follows:

Say we have the input: [1, 0, -1]

And weights and bias:
```
EXAMPLE

these are initialized randomly by nn.Linear(3,3)

W = [[ 1.0, 0.5, -1.0 ],  #   <----- weights for output neuron 0
     [ 0.0, 1.0,  0.5 ],  #   <----- weights for output neuron 1
     [-0.5, 0.5,  1.0 ]]

b = [0.0, 0.0, 0.0]
```

So in initialization, `layers` are hardcoded to have 6 layers and 5 
connections (connections via `nn.Sequential`). 

In the forwards pass, for each layer, we do a dot product between the input 
and the weights for that layer then apply GELU.

If shortcut is enabled, we also add the input to the result of the dot 
product.

In simpler terms, the dot product lets us see the changes between each layer 
while adding the original input back in lets us see the overall changes that 
are being made.

So using an example, without shortcut we see the following:

1. we make 10$
2. we make 10$
3. we lose 50$
4. we make 5$

We have no idea if we are in debt or not, is losing 20$ ALOT or not?

But with shortcut, we see this:

Initially we had 300$

1. we have 310$
2. we have 320$
3. we have 270$
4. we have 275$

> [!note]
> There is still no updating weights, so we basically just calculated the dot 
> product for each layer against input put it through GELU and added input 
> to the result and stored the final results in RAM.
>
> output = input + GELU(input × layer_weight + bias)

```python
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            # Compute the output of the current layer
            layer_output = layer(x)
            # Check if shortcut can be applied
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x


def print_gradients(model, x):
    # Forward pass
    output = model(x)
    target = torch.tensor([[0.]])

    # Calculate loss based on how close the target
    # and output are
    loss = nn.MSELoss()
    loss = loss(output, target)
    
    # Backward pass to calculate the gradients
    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            # Print the mean absolute gradient of the weights
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")
```

This code performs 1 forwards pass.

```python
layer_sizes = [3, 3, 3, 3, 3, 1]  

sample_input = torch.tensor([[1., 0., -1.]])

torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=True
)
print_gradients(model_with_shortcut, sample_input)
```


