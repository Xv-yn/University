# Bayesian Networks and Inference

## Basic Probability

- Marginal Probability: 
    - P(A) – the probability of A being true.

- Joint Probability: 
    - P(A, B) – the probability that both A and B are true simultaneously.

- Conditional Probability: 
    - P(A | B) – the probability of A given that B is true.

The full joint distribution lists the probabilities for every combination of 
variable values. For n binary variables, we need 2^n entries.

### Marginalization

To find the probability of a variable (e.g., Toothache), we sum over all 
entries where Toothache=true.

P(Toothache)=            ∑                 P(entry)
             all entries where Toothache=true

### Chain Rule

Used to decompose joint distributions into conditional ones: 

P(X1,X2,...,Xn) = P(X1)P(X2 | X1)P(X3 | X1, X2) ... P(Xn | X1,...,Xn−1)

### Independence

Two variables A and B are independent if: 

P(A,B) = P(A) x P(B)

### Conditional Independence 

A and B are conditionally independent given C if: 

P(A,B|C) = P(A|C) x P(B|C) 

This greatly simplifies the storage and computation of probabilities.

### Bayes' Theorem

Used for reverse reasoning:

P(A|B) = P(B∣A)⋅P(A) / P(B)

To compute P(B), we can marginalize over all possible causes: 

P(B) = ∑ P(B|Ai) x P(Ai)
       i

## Bayesian Networks (BNs)

Bayesian Networks are graphical models that encode probabilistic 
relationships using a Directed Acyclic Graph (DAG).

### Structure

- Nodes: 
    - Random variables
- Edges: 
    - Causal or influential relationships
- CPTs (Conditional Probability Tables): 
    - Probabilities for each variable given its parents

### Key Assumption

Each node is conditionally independent of its non-descendants, given its 
parents.

## Joint Probability with Bayesian Networks 

BNs allow us to compute the full joint distribution efficiently: 

                   n
P(X1,X2,...,Xn) =  ∏  P(Xi | Parents(Xi)) 
                  i=1

Example For a network with nodes A → B → C: 

P(A,B,C) = P(A) x P(B|A) x P(C|B)

## Inference in Bayesian Networks

- Inference by Enumeration 
    1. Convert conditional probability to joint. 
    2. Sum over hidden variables. 
    3. Normalize. 
    
    Drawback: Time complexity is O(n x 2^n)

- Inference by Variable Elimination 
    1. Identify factors (CPTs). 
    2. Multiply relevant factors. 
    3. Sum out hidden variables. 
    4. Normalize the result. 

    Improved efficiency by avoiding redundant calculations.

## Approximate Inference

Used when exact inference is too expensive.

Techniques
- Simple Sampling: Generate complete random samples.
- Likelihood Weighting: Weight samples by likelihood of evidence.
- Gibbs Sampling: A Markov Chain Monte Carlo (MCMC) method.

## Naïve Bayes as a Bayesian Network 

A Naïve Bayes classifier is a special case of a BN where: 
- The class is the parent node. 
- Features are conditionally independent given the class. 

P(Class|Features)∝ P(Class) x ∏ P(Feature_i|Class)
                              i

## Learning CPTs from Data

Estimating Probabilities 

- Unconditional: 
    
           Number of samples with B = True 
    P(B) = ---------------------------------
                    Total samples 

- Conditional (e.g., P(A | B, E)): 

                Number of samples where A,B,E are true 
    P(A|B,E) = ----------------------------------------
                      Number where B,E are true 

Data Required Each CPT entry is estimated using frequency counts from 
labeled data.


