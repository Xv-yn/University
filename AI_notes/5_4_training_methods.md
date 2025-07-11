# Temperature and Top-K Sampling

A common word that you'll hear when talking about LLMs is "temperature".

Temperature is a short hand way of saying "Temprature Scaling", which 
basically means dividing logits (see 4_7) by a values greater than 0.

By doing this we can achieve 1 of 2 results:
- Dividing by numbers > 1 will result in more uniformly distributed token 
  probabilities after softmax
- Dividing by numbers > 0 and < 1 will result in more confident (sharper/
  peaky) distributions after softmax

Here is a live example:



