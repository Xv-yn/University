# Descision Trees

This is basically the idea for a diagnosis algorithm.

Basically its as follows:
- If cough, go right, else, go left. 
- If sneeze, go right, else, go left.
- ...
- Until we get a disease.

```txt
          cough?
         /     \
      yes       no
     /           \
 sneeze?       headache?
  /    \         /     \
flu  cold   migraine  healthy
```

> [!note] NOTE
> Note that it doesn't necessarily have to be Yes/No, it could have multiple
> types of branches. Yes/No is just a easy to visualize example.

Think of it like the code behind the akinator if it helps.

## Building a Descision Tree

We can do this in a not efficient way (cuz this is important, no) or the 
efficient way.

We use a Top-Down recursive divide-and-conquer approach.

Lets assume we know everything about this dataset. Meaning that we know 
which attribute tells us the most information. 

For example, in cludeo where you have to guess the person, if half the people
wear glasses and the other other half don't, then we ask the question "Does
your character wear glasses", allowing us to remove half the board.

So once we pick the best attribute, we split that data bsed on that one 
atttribute.

And we repeat this process for each subset!

Now, by technicality  it could continue on forever, so there are a few 
stopping conditions:
- No need to split further — we already know the answer. Or
- Sometimes all examples have the same values for attributes, 
  but different class labels (due to errors or randomness).
- We’ve used up all the questions, but the examples still don’t 
  all belong to the same class.

# Entropy

Like in basic science, entropy is randomness. 

In this case, entropy is the measure of how random a dataset is. 
A simple example would be:
- Fair coin (50% heads, 50% tails):
    - Maximum uncertainty = HIGH ENTROPY
    - You have no idea what’s coming
- Rigged coin (90% heads, 10% tails):
    - Less uncertainty = lower entropy
    - You're more confident in guessing heads
- Double-head coin (100% heads)
    - No uncertainty = entropy = 0
    - You’re absolutely sure

The formula for calculating this entropy is as follows:

> [!note] NOTE
> M is the variable, so H(M) means the Entropy of variable M

H(M) = Entropy(M) = -(P_1 * log\_2(P\_1) + ... + P\_n * log\_2(P\_n))

P_1 is the probability of that variable occuring in that dataset.
For example in a dice roll (not rigged) P\_1 = 1/6

And there would be 6 of these "P_x * log\_2(P\_x)" added together.

Note that ALL probabilities must add up to 1.

## Information Gain

Information gain is a number that tells you how much entropy 
(disorder) is reduced when you split the data using a specific 
feature/attribute.

In simpler terms, information gain tells you how much that disorder 
drops after using an attribute to split the data.

As an example, lets say I want to know what disease someone has. I can 
ask "do you have a cough?". 

> [!note] NOTE
> Note that asking about a different variable (thats not the outcome) cuts
> the amount of data I need to look through.

As a result, this rules out other possible diseases that don't have cough
as a symptom. Thus, gaining me information (and reducing entropy)!

Now I can ask "Is your disease XXXX". This is not a good question because I'm 
going to be forced to ask for EVERY disease. In more "proper" terms, I'm 
reducing the entropy by too little of an amount (gaining too little information) 
in comparison to "filtering" by symptoms.

## Reminder

A reminder is the leftover disorder (entropy) after you split your data 
using an attribute.

It helps measure how effective your split was. The lower the reminder, the 
more helpful the attribute was — and the higher the information gain.

Information Gain = Entropy before split − Reminder

# Overfitting

Overfitting: the error on the training data is very small but the error on 
new data (test data) is high
- The classifier has memorized the training examples but has not extracted 
  pattern from them and is not classifying well the new examples!

A DT grows each branch of the tree deeply enough to perfectly classify 
the training examples. In ismpler terms This means the tree keeps growing 
and splitting until it can memorize every single example in the training 
data — even the weird or rare ones. 

It’s like a student who memorizes every question from the practice exam 
instead of learning the actual concepts.

This can be caused by having small training data or too much noise in the 
training data.

# Tree Pruning

We can fix this overfitting problem with tree pruining. There are two ways
we can prunt the DT:
- Pre-Pruning
    - Stops gorwing the tree earlier, before it reaches the point where it 
      perfectly classifies data
- Post-Pruning
    - Fully grows the tree then prune it

## Sub-tree Replacement
Replace an entire sub-tree (branch) with a leaf node.
- The new leaf predicts the majority class of the examples in that sub-tree.
- It’s like saying: “This whole part is too specific — let’s just generalize 
  it to a single answer.”

## Sub-tree Raising
Move a lower sub-tree up, replacing the higher node and its siblings.
- Useful if the lower branch captures the pattern better than the upper part.

## Rule Pruning
Convert the entire decision tree into a list of if-then rules (like 
rule-based logic).
- Then, prune (simplify) the rules one by one — removing unnecessary 
  conditions.
- This can sometimes give better results and more human-readable output.

## Stop Condition for Pruning

We can stop pruning by estimating accuracy of the model using the validation 
set or the traiing data (but more realistically teh validation set).

## Dealing with Numeric Attributes

In the case of having numeric attributes, we could basically have an infinite 
number of branches. So what we do is simply something like "is greater than 
75". 

Or we can set it such that its like "between 0 and 33", "between 34 and 66" 
and "between 67 and 99".

Simply turning numerical attributes to categorical.

## Gain Ratio

Gain Ratio is a modified version of Information Gain that: 
- Still tries to find attributes that reduce entropy (like info gain), 
- But it penalizes attributes that split the data into too many branches.

SplitInformation(S|A) = - \SUM (|S_i|/|S| log_2 (|S_i|/|S|))

S = original dataset 
S_i = a subset of the data after the split using attribute A

- Computing the entropy of the distribution of sizes of these subsets

GainRatio(S|A) = InformationGain(S|A)/SplitInformation(S|A)


