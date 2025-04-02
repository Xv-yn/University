# Baye's Theorem

P(A|B) = P(B|A) * P(A) /
              P(B)

P(A|B) = Probability of A given B
P(B|A) = Probability of B given A
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





