# Baye's Theorem

P(A|B) = P(B|A) * P(A) /
              P(B)

P(A|B) = Probability of A given B
P(B|A) = Probability of B given A

> [!important]
> Note that P(B|A) can be split into:
> - P(Variable_1 = XXX | A)
> - P(Variable_2 = XXX | A) and so on...
> 
> And P(B|A) = P(Variable\_1 = XXX | A) * P(Variable\_2 = XXX | A) * ...

P(A) = Probability of A
P(B) = Probability of B

## Example

Let's say:
- 1% of people have the disease (P(Disease) = 0.01)
- The test is 99% accurate:
    - If you have the disease: P(Pos|Disease) = 0.99
    - If you don't have the disease: P(Pos|NoDisease) = 0.01

We want to find: what are the chances we have the disease if we test positive?
P(Disease|Pos)

Using Law of Total Probability:
P(Pos) = P(Pos|Disease) * P(Disease) + P(Pos|NoDisease) * P(NoDisease)
       = 0.99           * 0.01       + 0.01             * 0.99
       = 0.0198

Using Baye's Theorem:
P(Disease|Pos) = P(Pos|Disease) * P(Disease) = 0.99 * 0.01 = ~0.5
                           P(Pos)                 0.0198 

From this, we've determined that if we test positive, we have a 50% chance of having
the disease.

> [!note] NOTE
> Look back at how common the disease is

If the [!NOTE]  didn't help, this is very unintuitive, so let's break it down.

Let's say out of 10,000 people:
- Only 1% have the disease - 100 people
- 99% don't have the disease - 9,900 people

The test:
- Catches 99% of true cases (test correctly) - 99 people
  - In other words, 99% of healthy people test negative (test correctly) - 9,801 people
  - Meaning 1% of healthy people test positive (test incorrectly) - 99 people
- Misses 1% of true cases (test incorrectly) - 1 person

So looking at the tests:
- 99 people test Positive (True Positive)
    - 9,801 people test Negative (True Negative)
    - 99 people test Positive (False Positive)
- 1 person tests Negative (Fase Negative)

Summing this up we have a total of 99 + 99 + 1 + 9,801 = 10,000. Which tracks.

Meaning that if I test positive, I am either 1 of the 99 people who have the disease
OR I am 1 of the 99 people who do not have the disease.

Hence, I have a 50% chance of having the disease given that I test positive.

# Dataset Problems

## Laplace Corection
Now, say a value doesn't appear. For example, there are no people who test postive.
As a result, this will cause any form of prediction to always say they test negative.
Hence, we just add 1 to the top and k (k being the number of possible outcomes) to the
bottom.

P(A|B) = P(B|A) * P(A) + 1 /
              P(B) + k

## Missing Values

In the case there are missing values in the dataset, we can just ignore them.
But if we don't want to do that, we can find ways to estimate those values. Methods
include those learned before, K Nearest Number, Model it as another value, etc.

## Numeric Attributes

In our disease example, we deal with boolean/categorical attributes, "positive" and 
"negative".


# Evaluating Classifiers

> [!note] RECAP:
> A classifier is a type of algorithm that assigns input data to a predefined
> category or class. Like MNIST that "classifies"/reads a drawing and 
> turns it into a number.

The general issue is that say I've built a classifier that can recognize 
many different things, os how would I know if it can identify them correctly? 
It would be too troublesome to test each item 1 by 1 manually.

## Holdout Procedure

This is the most basic method for testing classifiers. Given a dataset, we
split it such that 2/3 of the data would be used for training, while the last
1/3 is used for testing.

This works intuitively, where the process of this training is:
1. Data is split into training and testing
2. Model trains on training data
3. model tests itself on the test data and outputs its accuracy

### Extention of Holdout Procedure (Validation Set)

By adding a validation set, it allows us to fine tune the parameters. This 
basically means we create a "mock test" for the algorithm.

Thus, we split that data into 60% training data, 20% validation
data and 20% testing data.

The general process is as follows:
1. Model trains on training data
2. Model tests itself on the validation data
3. Model checks if the result of the validation test is good enough
4. If not good enough, modify some parameter and repeat from step 1, 
   else, test on testing data.

### Stratification

Stratification is basically making sure both the training and test sets
have a similar distribution of classes.

In simpler terms, say we have "Is Chair" and "Is Not Chair". If the training 
data contains many "Is Chair" and only 1 "Is Not Chair", the model will 
get everything wrong!

Stratification splits the dataset while preserving the proportion of 
classes, ensuring that the training, validation, and test sets all have a 
similar distribution of each class.

Basically it means that training data will be 50% "Is Chair" and the other 
50% "Is Not Chair". Similarly, the validation and testing data will also
comprise of 50% "Is Chair" and 50% "Is Not Chair".

This is especially important when separating data into training, validation 
and testing data. 

### Repeated Holdout Method

This is literally what it sounds like.

We literally just repeat the following process a few times:
1. Split the data into training and testing (random for each iteration)
2. Train model based on training data
3. Test model based on testing data nd get accuracy of model

> [!note] NOTE
> Due to the random splitting, there is no guarantee that every data point 
> will end up in a test set at least once.

```txti
First Iteration:
+-------------------------------+------+--------+
| ///////// train ///////////// | TEST | ////// |
+-------------------------------+------+--------+

Second Iteration:
+------+------+---------------------------------+
| //// | TEST | /////////////////////////////// |
+------+------+---------------------------------+

Third Iteration:
+----------------------+------+-----------------+
| //////////////////// | TEST | /////////////// |
+----------------------+------+-----------------+
                                        <------->
Note that this section is never used in tests ^
```

Something to note is that the model does not learn from the previous iteration.
The goal is to prevent things like "a lucky split (just so happens that the 
testing data is easy)"

## Cross Validation

This is DIFFERENT from the repeated hold out method

The data is systematiclaly split into x sections. Then each section is then 
used for testing. While the rest are used for training. As a result, there 
would be x iterations.

So lets say we split the dataset into 10 sections. We then use the first 
sections for testing, and the last 9 sections for training. In the second 
iteration we use the second section for testing and 1st and 3rd to 10th 
section for training and so on.

Same as Repeated Hold Out Method, this does not let the model learn from
previous iterations, this is just to see how accurate the model really is.

```txti
First Iteration:
+-------+-------+-------+-------+-------+-------+
| TEST  | train | train | train | train | train |
+-------+-------+-------+-------+-------+-------+

Second Iteration:
+-------+-------+-------+-------+-------+-------+
| train | TEST  | train | train | train | train |
+-------+-------+-------+-------+-------+-------+
```

### Leave-One-Out Cross Validation

This is the same as regular cross validation BUT instead of breaking the 
data into sections, we split it by data points.

In simpler terms, say we have 100 data points. We use 99 points as training
and the last point as testing. And repeat for each point. This leads to 
100 iterations of testing.

This results in a bunch of 100% accurate tests and 0% accurate tests. Thus,
we average to see how many we got right out of the 100 iterations.

## Comparing Classifiers

Now that we have two methods to determine the accuracy of a classifier model,
then we want to know which method is better!

> [!note] NOTE
> We can use any statistical method that is VALID for the data

In this case, we use a paired t-test. The way we structure it is as follows:
```txt
        | Repeated Hold Out Method | Cross Validation |
        +--------------------------+------------------+
Fold 1  |            95 %          |        90 %      |
Fold 2  |            90 %          |        98 %      |
Fold 3  |            94 %          |        92 %      |
...     |            ...           |        ...       |
Fold 10 |            91 %          |        95 %      |
--------+--------------------------+------------------+
 mean   |            94 %          |        96 %      |
```
Using a paired t-test, we can get the p-value.

If the p-value is > 0.05 then we CANNOT tell if one method is better than the
other. 

However, if the p-valuue < 0.05, the we KNOW THAT one method is better.

Or we can skip that all together and use the confidence interval.

## Confusion Matrix

A confusion matrix is a summary table used to evaluate how well a 
classifier performs, especially on classification tasks. 

It shows the counts of: 
- Correct predictions 
- Mistakes

Here is an example of a confusion matrix showing true/false positives/negatives.
```txt
                Actual
              Yes    No
Predicted  ----------------
   Yes     |  TP  |  FP  |
   No      |  FN  |  TN  |

TP = True Positive
FP = False Positive
FN = False Negative
TN = True Negative
```

### Calculating Performance Measures

Precision = TP / (TP + FP)

Precision is of predicted Yes, how many are actually Yes.
In simpler terms, how “trustworthy” are positive predictions?

Recall = TP / (TP + FN)

Recall is of actual Yes, how many were predicted Yes.
in simpler terms, how many actual positives did we catch?

F1_Measure = (2 * Precision * Recall) / (Precision + Recall) 

F1 Measure is balance between precision & recall

### Cost Based Evaluation

It’s an evaluation method where we assign different costs to different 
types of errors, and measure model performance based on total cost, 
not just error count.

Outcome	Meaning	Cost:
- TP	Correct positive	$0
- TN	Correct negative	$0
- FP	False alarm	        $1
- FN	Missed detection	$100

In this scenario, its like assigning weights to tell us how important a
false positive or missed detection is.

# Terminology

- Accuracy = How many CORRECT classifications per 100

- Error Rate = How many WRONG classifications per 100





