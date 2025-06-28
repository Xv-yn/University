# Data Sampling

To train a model, we can use the methods learned in COMP3308, such as K-fold-
crossvalidation, or Leave One Out Cross Validation, splitting the dataset 
into 3 sections, training, validation and testing, etc.

In the case of word generation, LLMs are trained to generate 1 word at a time.
Like so:
```
Evidence:                   ---->  Prediction:
-------------------------------------------------
 and                        ---->  established
 and established            ---->  himself
 and established himself    ---->  in
 and established himself in ---->  a
```

More formally:
```
 inputs = tensor([["In",  "the",     "heart", "of"  ],
                  ["the", "city",    "stood", "the" ],
                  ["old", "library", ",",     "a"   ],
                  [...                              ]])

targets = tensor([["the",     "heart", "of",  "the"   ],
                  ["city",    "stood", "the", "old"   ],
                  ["library", ",",     "a",   "relic" ],
                  [...                                ]])
```

Using the first line as an example:
```
Input:                      ---->  Output:
-------------------------------------------------
 In the heart of            ---->  the heart of the city
```

> [!note] Note:
> Remember that instead of using words, we must tokenize the text and convert
> it into integers

In code:
```python
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt)

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize the tokenizer
    tokenizer = get_encoder(model_name="gpt2_model", models_dir=".")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader
```
