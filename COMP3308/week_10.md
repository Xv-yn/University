# Support Vector Machines (SVM)

## Maximum Margin Hyperplane

Given  the following graph:
```txt
 ^
 |                  o      o         
 |x                             o 
 |                         o      
 |                            o  o  
 |  x  x                   o       
 |        x                     o   
 | x          x                
 |   x  x        x               
 +---------------------------------->
```

We can add a linear liner to separate the data. However, there are many options, 
and we want to find the "best" line!

```txt
   
    <-- Margin -->
 ^  
 |  '.    '.     '.(o)      o         
 |x   '.    '.  /  '.           o 
 |      '.    '.     '.    o      
 |        '. /  '.     '.     o  o  
 |  x  x    '.    '.     '.(o) <------------ Support Vector     
 |        x   '.    '.     '.   o   
 | x         (x)'.    '.     '.
 |   x  x      (x)'.    '.     '.
 +-----------------------^---------->
                         |
                      Boundary
```

The entire area of the the `Margin * Y-Axis` is called the hyperplane

The boundary is in the middle of the margin

Support vectors are the examples (data points) that lie closest to the 
decision boundary (they are circled)
- The support vectors just touch the margin of the decision boundary
- It is possible to have more than 1 support vector for each class
- For our example: 4 support vectors, 2 for class `x` and 2 for class `o`
- If we move another input example, that is not a support vector, the decision 
  boundary will not change

Margins are the separation between the boundary and the closest examples
- The bigger margin is more likely to classify more accurately new data

The hyperplane with the biggest margin is called the maximum margin hyperplane 
(separator)

SVM selects the maximum margin hyperplane

Mathematically, the boundary is denoted with H and the upper and lower boundaries
are denoted with H₁ and H₂.

Where:
- H : w * x + b =  0 (Boundary)
- H₁: w * x + b =  1 (Upper)
- H₂: w * x + b = -1 (Lower)

> [!note]
> w is the same for H, H₁and H₂. What varies is b!

Now this is where it gets confusing.
- H = 0 does not mean that the Y axis is 0. 0 is a relative number used to 
  represent that this is the base
- H₁ = 1 means that it is 1 unit above H. but doe NOT specify what that unit is.

To officially calculate the distance between H and H₁ we use the following 
formula:

d = 1 / ||w||

Where ||w|| is the euclidean norm (vector magnitude) of the weight of the vector 
H. Hence, the full margin  can eb calculated as:

d =  2 / ||w||

## Kernel Trick

In practice most problems are linearly non-separable

SVM with soft margin will find a linear boundary that optimizes the
trade-off between the errors on training data and the margin

SVM can further be extended to find non-linear boundary

In Support Vector Machines (SVMs), the Lagrange multipliers λ_i are introduced 
during the optimization of the dual problem. Their values tell us something 
critical about the data points.

If λ_i = 0, then example i does not influence the decision boundary. 
- It is not a support vector.

If λ_i > 0, then example i is a support vector, i.e., it lies on the margin or 
violates it slightly (depending on the setting).

Remeber that SVM is still a ML algorithm. Hence, during training we compute dot 
products between training vectors ONLY.

# Ensemble Classifiers

Exactly as it sounds like, we grab a bunch of classifiers and "ensemble" them
(have them run together/in parallel with the same input) and pick the best 
answer/mix the answers to form the perfect answer.

## Bagging

When we have a bunch of solutions, we leave it to a majority vote. This is 
called 'bagging'. 

In bagging the ensemble members are built separately

## Boosting

The 'boosting' method assigns weights to each problem solver, e.g. if person 
X has solved 999 problems with a 90% accuracy and person Y has solved 999 
problems with 40% accuracy, we take the majority of person X's solution and 
some of person Y's solution.

In boosting the ensemble members are built iteratively – the new ensemble 
members are influenced by the performance of the previous ones

## Random Forest

1) Create multiple (smaller) subsets of features from the original feature
set by randomly selecting the features

2) The result is a multiple versions of the training data, each containing
only the selected features

3) Build a classifier for each version of the training data

4) Combine predictions with a majority vote

### Simpler Terms

1. Take Random Samples of Data
    - Make many different random copies of your training data.
    - Each copy has some of the original data, picked randomly (some rows may 
      repeat!).

2. Pick Random Features
    - When each tree is deciding how to split (ask questions), it only looks 
      at some of the features, not all of them.
    - This makes each tree a little different.

3. Build a Bunch of Decision Trees
    - Train one tree on each random sample of data.
    - Each tree makes its own decision on new inputs.

4. Combine Their Answers
    - When it’s time to make a prediction:
    - If it’s classification (e.g., yes/no): use majority vote.
    - If it’s regression (e.g., numbers): take the average of all tree 
      predictions.

