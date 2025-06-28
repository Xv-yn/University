# Tokenization

The essence of tokenization is converting text to integers

We represent "knowledge" that a computer can understand into vectors.

```
Second Dimension
       ^
       |       +---------------+
       |       | Eagle         |<------Vector embedding of different
       |       |   (x)   Goose |       types of birds
       |       |    Duck   (x) |
       |       |     (x)       |
       |       +---------------+
       |
       |                                    Squirrel
       |           Berlin                     (x)
       |   Germany   (x)                       ^
       |     (x)         London                |
       |       England     (x)                 Vector embedding of the word
       |         (x)                           "Squirrel"
       +-------------------------------------------------------------->
                                   First Dimension
```

The essence of tokenization is as follows:
1. Determine a vocabulary
2. Assign a unique int to each item in the vocabulary (like a dictionary)
3. Done!
    - When Tokenizing a text, refer back to the "dictionary" to convert a 
      text into tokens
    - Same vice verca

## Detailed Steps

This requires an already existing text. For example, we could get a `.txt` of 
the Harry Potter series.

We then split all words and symbols in the text to get something like this:
```
['Hello', ',', 'world', '.', 'Is', 'this', '--', 'a', 'test', '?']
```

Afterwards, we remove duplicates (turn it into a set) and assign a token 
(integer) for each unique word.
```
vocab = {token:integer for integer,token in enumerate(all_words)}
```

Here is a class object:
``` python
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
                                
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
```

We can also add special tokens such as `|unk|` for when we get a word that is 
outside our known vocabulary (to handle new zoomer terms that have yet to be 
created).

## Realistic Usage

Now in most cases, we will not create our own tokenizer as it takes way too 
long and requires quite a bit of training. This is because a more efficient 
tokenizer uses Byte Pair Encoding (BPE).

BPE is basically splitting a word into smaller subgroups like so:
```
                 thought
                 /     \
             thou       ght
             /  \       / \
           th    ou   gh   t
           |     |    |    |
TokenID: 3301    122  37   5
```

As such we already have GPT-2's BPE tokenizer, so we can just use that.

Here is a sample implementation that uses OpenAI's BPE GPT-2 tokenizer:
```python
from bpe_openai_gpt2 import get_encoder

# Load tokenizer
encoder = get_encoder(model_name="gpt2_model", models_dir=".")

text = "Hello, I am building a tokenizer from scratch!"

encoded = encoder.encode(text)
print("Encoded tokens:", encoded)

decoded = encoder.decode(encoded)
print("Decoded text:", decoded)
```

