# Embeddings

We will create what we call an embedding_layer. Where it initializes random 
values into the context (we can determine the number of context) for 
each word (properly called, token). This causes the embedding_layer to 
look like the following.
```
embedding_layer = {
    "1": [0.1, 0.6, 0.2, 0.1], # 1 "Hello"
    "2": [0.1, 0.1, 0.8, 0.0], # 2 "I"
    "3": [0.3, 0.2, 0.2, 0.2], # 3 "like"
    "4": [0.7, 0.1, 0.1, 0.1], # 4 "books"
    "5": [0.1, 0.3, 0.3, 0.3], # 5 "and"
    "6": [0.2, 0.3, 0.4, 0.1]  # 6 "trains"
}
```

For ease of understanding, here is a wrong but similar version the 
embedding_layer:

```
vocabulary = {
     "Hello": {"Happy": 0.1, "Angry": 0.6, "Sad": 0.2, "Neutral": 0.1}
         "I": {"Happy": 0.1, "Angry": 0.1, "Sad": 0.8, "Neutral": 0.0}
      "like": {"Happy": 0.3, "Angry": 0.2, "Sad": 0.2, "Neutral": 0.2}
     "books": {"Happy": 0.7, "Angry": 0.1, "Sad": 0.1, "Neutral": 0.1}
       "and": {"Happy": 0.1, "Angry": 0.3, "Sad": 0.3, "Neutral": 0.3}
    "trains": {"Happy": 0.2, "Angry": 0.3, "Sad": 0.4, "Neutral": 0.1}
}
```
Note that in the wrong example, it shows the context, in this case "emotion",
and it's randomly initialized value (because the model does not understand 
"emotions" yet) for each token.

Here is another visualization in 2 Dimensions:
```
      ^
      |         [0.6, 0.1]
Happy |           /
      |         x
      +------------>
           Angry
```
Think of each "emotion" as another dimension. So if we add "sad", we get:
```
      ^   ^ Sad
      |  /      [0.6, 0.1]
Happy | /        /
      |/        x
      +------------>
           Angry
```

Positonal Embeddings allow for the the model to understand and perceive 
grammar. They MUST have the SAME number of dimensions as the embedding 
layer. If not, then we could use vector projection (if we are adding 
token_embedding to positional_embedding to produce the input_embeddings) or 
concatenation or some other method.

```
     Input Embeddings: [0.15, 0.3, 0.25]   [0.31, 0.13, 0.22]
                               ^                    ^
                               |                    |
                               |                    |
Positional Embeddings: [0.05, 0.1, 0.00]   [0.11, 0.03, 0.20]
                               +                    +
     Token Embeddings: [0.15, 0.2, 0.25]   [0.20, 0.10, 0.02]
                             ^                   ^
                        Token 1                 Token 2
```

Similarly, at the start, they are randomly initialized and are changed via 
training.
