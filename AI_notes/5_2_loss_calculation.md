# Calculating Loss

Right before we calclate loss, we need to turn the raw logits into 
probabilities.

Simply put, we apply softmax to each row of the logits matrix.

```
Visualization of logits:
        ^      ["every"   [0.1,0.6,0.1,0.2],
Tokens  |       "pokemon" [0.2,0.4,0.1,0.3],
        v       "likes"   [(0.5),0.3,0.0,0.2]]
                             |    
                             v
                          index: 0
                     We take this one  because (0.5 > all other values) and 
                     reference the vocabulary o grab the word
                           
                           <-------------->
                    Relation in terms of probabilities
                      with each word in vocabulary

NOTE: These values don’t represent probabilities yet — just raw “preferences” 
the model has for each vocabulary word.
```

In doing this, we take the index with the highest probability to predict the 
next word.

The idea is: to predict the next word, we take the word with the highest 
probability of occurring.

Process so far:
1. Grab Logits
2. Softmax Logits (turns logit values into probabilities)

Now, with untrained weights, we know that the current predictions are wrong. 
But the model needs to know HOW wrong it is. To do this we need to calculate 
loss.

So, we grab the predicted probability assigned to the correct word.

For example:
If the target next word is “zoo” (say, at index 0), we take:
```python

predicted_probs[target_idx] → 0.007454
```

In essence, this is saying, we want to predict the word "zoo", but it only 
has a 0.7% chance of appearing (when it should be higher because it is the 
correct word).

We repeat this for the other tokens in the sequence.

Process so far:
1. Grab Logits
2. Softmax Logits (turns logit values into probabilities)
3. Grab the Target Probabilities (grab probabilities of the correct words)

If we want to measure how good a sequence is (e.g., "every effort moves"), 
you could multiply the predicted probabilities of each word in the sequence:
```
P(word1) × P(word2) × P(word3) ...
```

But that becomes a very small number, hard to work with and unstable in 
floating point.

Hence, we use log probabilities convert multiplication → addition
By taking the log of each probability:
```
log(P1 × P2 × P3) = log(P1) + log(P2) + log(P3)
```

This makes it easier to compute and more stable during optimization.

For example:
If the model assigns the correct word a probability of 0.0001, then:
```
log(0.0001) ≈ -9.21
```
So the model gets penalized. But if it predicts 0.9, then:
```
log(0.9) ≈ -0.10
```

In this scenario, we've turned the probabilities into a value where the 
further away from 0, the bigger the error. Where the ideal scenario would be:
```
log(1) = 0.
```
Where `1 = 100%` and `1` represents the probability of choosing the correct 
word.

Process so far:
1. Grab Logits
2. Softmax Logits (turns logit values into probabilities)
3. Grab the Target Probabilities (grab probabilities of the correct words)
4. Apply Log to all Target Probabilities

We now know how wrong the predictions are (by knowing how much our 
probabilities of the correct token are off by). But, there are multiple and 
we need to summarize how off our predictions are.

To do so, we average these log values, so we simply take the mean.

Process so far:
1. Grab Logits
2. Softmax Logits (turns logit values into probabilities)
3. Grab the Target Probabilities (grab probabilities of the correct words)
4. Apply Log to all Target Probabilities
5. Take the Mean of All log(Target Probabilities)

And we now have a negative mean, which makes sense. But for easier 
understanding and convention, we multiply by -1 to turn it positive to 
intuitively show that we have a chunk of loss rather than we "owe" values.

Process so far:
1. Grab Logits
2. Softmax Logits (turns logit values into probabilities)
3. Grab the Target Probabilities (grab probabilities of the correct words)
4. Apply Log to all Target Probabilities
5. Take the Mean of All log(Target Probabilities)
6. Multiply by -1

> [!note]
> This entire process is also known as CrossEntropyLoss Calculation which has 
> a separate function in the `nn` library

> [!important]
> In most cases, you will have to flatten the matrix by the batch dimension 
> so that PyTorch can run the `cross_entropy()` function

In python it is implemented like so:
```python
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches
```
