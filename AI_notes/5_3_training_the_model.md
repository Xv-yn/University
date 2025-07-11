# Training the Model

To train the model, we follow the following steps:

> [!note]
> We are using the validation training method where we split the data into 
> 3 parts, training (80%), validation (10%) and testing (10%)

- For each epoch
    - For each batch in the training set
        - Set gradients to 0 (to remove any influence from last training batch)
            - Gradients are an attribute of the weights
            - Think of weight matrices as a class that has 3 main attributes, 
              the dimensions (shape), the weight values, and the gradients
        - We calculate the loss using the methods and functions learned 
          before (see 5_2)
        - We calculate the loss gradients via `loss.backwards()`
            - Using backpropagation we calculate the gradients of the loss
                - This essentially means that the model determines which 
                  weights contribute the most to the loss
            > [!note] Refresher on Backpropagation
            > Calculates output using input, if off by < 80% then readjusts 
            > weights and recalculate output until ouput is within 80% of 
            > expected ouput
        - Now that we know which weights caused the most problems, we update 
          those weights via `optimizer.step()`
          - We update these weights in a similar way we update perceptron 
            weights, but using a more complex equation (some kind of 
            momentum calculation)
        - We then record 3 pieces of data that would be helppful for model 
          analysis and does not have direct relation towards model learning
          - The tokens the model has seen, overall training loss and overall 
            validation loss

> [!note]
> `model.train()` toggles the model's internal behavior to enable things like 
> dropout
> `model.eval()` disables dropout and other randomness to make evaluation deterministic
> `torch.no_grad()` tells the model to avoid calculating gradients, saving 
> memory and computation

The below is an implementation of simple training of a model

```python
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)
```

> [!note]
> AdamW is an improved version of Adam (Adaptive Moment Estimation) with 
> proper weight decay. Simply put, its just a more complex version of a 
> perceptron neural network, adding things like means of gradients (gradients = 
> change in weights), etc. into the calculation

