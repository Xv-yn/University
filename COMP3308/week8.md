# Introduction to Neural Networks

These work on the basis of a biological neuron.

Where we send data from one end, it goes through the cell body (which changes
the data somehow), and the data is sent out.

This is an ASCII representation of a SINGULAR artificial neuron.

```txt
----->-.
IN >>   \    
         \.-'''-.
         /       \
----->--(         )------>-
IN >>    \       /  OUT >>
         /'-...-'
        /
----->-'
IN >>
```

An artificial neural network is simply LOTS of singular aritificial neurons
that are connected in some way.

If we think of a neural network in the form a graph, each edge of the graph
is called a `connection` and each connection has a `weight`. While each node
is a singular neruon.

In most cases a neural netowrk is presented in a directed graph.

Each neuron:
- Recieves a NUMERIC input
- Produces a NUMERIC output

A neural network can be simplified into 3 main layers:
- Input Layer
- Hidden Layer
- Output Layer

A summary of how each neuron works can be put into a simplet equation:

final_output = custom_function(input: SUM (weight*input_values + bias))

In "english", we multiply the weights by the input values and add bias. This
is a way to summarize all the inputs into the neuron. Then we apply some 
custom function to modify this value to give us a final output.

# Types of Neural Networks

There are 3 main types of neural networks
- Feedforward supervised
    - Info flows one way: input → hidden layers → output
    - No loops or cycles
    - You train with labeled data (you know the right answers)
- Feedforward unsupervised
    - You don’t have labeled answers
    - Used to find patterns, clusters, or features in raw data
- Recurrent
    - These networks have loops (they feed data back into themselves)
    - Used for sequences and time-related tasks (like speech, text, video, 
      or anything temporal)

# Perceptron

A perceptron is a single neuron that makes a binary decision — like yes 
or no, 0 or 1.

It works in the same way as a regular neuron but the step function, `f()`, 
determines if the output is a "yes" or "no".

If we were to do this mathematically, we would observe that it creates a 
linear boundary that splits space into “yes” and “no” regions.

This is linear because we are calcualting a weighted sum followed by a binary 
step function. 

# How does it learn?

Given that we have a random line, and we have lines that we want that random 
line to match. We compare that line and against another and we get a 1 or 0. 
As a result, we modify the weights slightly to move the original line closer 
to what we want. And repeat!

In a more formal approach:
- If target t=1, output a=0:
    - The model predicted 0, but it should've said 1.
    - So we add the input to the weight:
        w_new = w_old + input
- If target t=0, output a=1:
    - The model predicted 1, but it should've said 0.
    - So we add the input to the weight:
        w_new = w_old - input
- If target = output, t=a:
    - The prediction is already correct, so:
        w_new = w_old

> [!note] Note
> p = input
> t = target
> a = output
> w = weight
> b = bias
> e = error

```txt
Init:
w = [0,0,0]
b = 0

Each Iteration:
p = [0,0,0]     t = 0
a = step((p * w) + b)
e = t - a
w_new = w_old + (e * p)
b_new = b_old + e
```
