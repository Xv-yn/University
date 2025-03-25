# Machine Learning

There are 3 main types of computer learning:
- Supervised
- Unsupervised
- Reinforcement

A fourth type (which is very old) is called association learning

## Supervised Learning

In a sense, its name is intuitive. The data that the model learns from must
be from an existing dataset. It doesn't learn from anything outside that
dataset.

And in most cases, that dataset is "supervised" by a human.

There are 2 main types of supervised learning:
- Classification
    - This is identifying something, like facial recognition, the MNIST 
      number thing
    - Usually categorical
- Regression
    - This is using past data to predict future data 
    - Usually numerical

## Unsupervised Learning (Clustering)
Given a collection of data, we let the computer find ways to group them.
Points that generally "cluster" to a specific area can be grouped.


## Reinforcement Learning
This is basically smacking the computer everytimes it gives a bad output until
it learns to give a good output.

But instead of "smacking" its more like giving scores for specific outputs
and telling it to do better.

## Supervised Learning (Classification) K-Nearest Neighbor Algorithm

This algorithm involves a massive space complexity as it records and remembers each and
every piece of training data.

A summary of this algoritm is basically:
- Remember the training data
- When an input is recieved:
    - It searches the remembered training data
    - It finds the closest point/piece of data that is the most similar to the input data
    - It classifies the input data as the same

An example would be:
- Say you have a person who scores 80% on average for all their tests
- Now you have a person who has taken the same tests but has missed one, but has 
  scored 80% average as well
- Thus, using this algorithm we say that since these two people have scored basically the same
    -> the person who has missing data can be filled in as 80% 

## Normalization

In some cases when we are comparing more than 1 variable against each other. If we calculate
using manhatten distance, the data would be skewed.

To prevent this we normalize the data such that the greatest value for each variable is equal
to 1 and the lowest is equal to 0. And any value in between is some decimal between 0 and 1.
As a result of this, this allows for each variable to be weighted equally.

In terms of weightage, we can apply a weightage as needed.


