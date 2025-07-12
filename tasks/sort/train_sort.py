import torch
from transformer import SimpleTransformer
from trainer import Trainer, SequenceDataset, create_vocabulary, generate_test_sequences
from torch.utils.data import DataLoader

# Configuration
vocab_size = 10
seq_length = 8
d_model = 64
num_heads = 4
num_layers = 2
batch_size = 32
num_epochs = 30

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab = create_vocabulary(vocab_size)

# Prepare data for the sort task
train_dataset = SequenceDataset(vocab_size, seq_length, 1000, task='sort')
val_dataset = SequenceDataset(vocab_size, seq_length, 200, task='sort')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Create and train the model
model = SimpleTransformer(
    vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    max_seq_length=seq_length
)
trainer = Trainer(model, device)
trainer.train(train_loader, val_loader, num_epochs, save_path='model_sort.pth')

# Test on a few examples
test_data = generate_test_sequences(vocab_size, seq_length, 5, task='sort')
model.eval()
for input_seq, target_seq in test_data:
    input_seq = input_seq.unsqueeze(0).to(device)
    with torch.no_grad():
        output, _ = model(input_seq)
        pred = torch.argmax(output, dim=-1)[0]
    print("Input: ", [str(i.item()) for i in input_seq[0]])
    print("Target:", [str(i.item()) for i in target_seq])
    print("Pred:  ", [str(i.item()) for i in pred])
    print()

# Interactive demo
print("Try your own sequence! Enter 8 numbers separated by spaces (e.g., '1 2 3 4 5 6 7 8')")
print("Type 'quit' to exit.")
while True:
    user_input = input("Enter sequence: ").strip()
    if user_input.lower() == 'quit':
        break
    try:
        tokens = [int(tok) for tok in user_input.split()]
        if len(tokens) != 8:
            print("Please enter exactly 8 numbers.")
            continue
        input_seq = torch.tensor(
            tokens, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            output, _ = model(input_seq)
            pred = torch.argmax(output, dim=-1)[0]
        print("Predicted sort:", [str(i.item()) for i in pred])
    except Exception as e:
        print("Error:", e)
