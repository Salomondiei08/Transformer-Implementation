import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import random
from transformer import SimpleTransformer


class SequenceDataset(Dataset):
    """
    Dataset for sequence tasks like copy, reverse, etc.
    """

    def __init__(self, vocab_size, seq_length, num_samples, task='copy'):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_samples = num_samples
        self.task = task
        self.data = self._generate_data()

    def _generate_data(self):
        """Generate training data based on the task."""
        data = []

        for _ in range(self.num_samples):
            # Generate random sequence
            input_seq = torch.randint(1, self.vocab_size, (self.seq_length,))

            if self.task == 'copy':
                # Copy the sequence
                target_seq = input_seq.clone()
            elif self.task == 'reverse':
                # Reverse the sequence
                target_seq = torch.flip(input_seq, dims=[0])
            elif self.task == 'sort':
                # Sort the sequence
                target_seq = torch.sort(input_seq)[0]
            elif self.task == 'shift':
                # Shift sequence by 1 position
                target_seq = torch.roll(input_seq, shifts=1, dims=0)
            else:
                raise ValueError(f"Unknown task: {self.task}")

            data.append((input_seq, target_seq))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Trainer:
    """
    Training utility for Transformer models.
    """

    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=0)  # Ignore padding token
        self.optimizer = None
        self.scheduler = None

    def setup_optimizer(self, lr=0.001, weight_decay=1e-4):
        """Setup optimizer and scheduler."""
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=30, gamma=0.1)

    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(dataloader, desc="Training")

        for batch_idx, (input_seq, target_seq) in enumerate(progress_bar):
            input_seq = input_seq.to(self.device)
            target_seq = target_seq.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output, _ = self.model(input_seq)

            # Reshape for loss calculation
            output = output.view(-1, output.size(-1))
            target = target_seq.view(-1)

            # Calculate loss
            loss = self.criterion(output, target)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        return total_loss / num_batches

    def evaluate(self, dataloader):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for input_seq, target_seq in dataloader:
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)

                # Forward pass
                output, _ = self.model(input_seq)

                # Reshape for loss calculation
                output = output.view(-1, output.size(-1))
                target = target_seq.view(-1)

                # Calculate loss
                loss = self.criterion(output, target)
                total_loss += loss.item()

                # Calculate accuracy
                predictions = torch.argmax(output, dim=-1)
                correct_predictions += (predictions == target).sum().item()
                total_predictions += target.size(0)

                num_batches += 1

        avg_loss = total_loss / num_batches
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        return avg_loss, accuracy

    def train(self, train_dataloader, val_dataloader, num_epochs,
              save_path=None, early_stopping_patience=10):
        """Train the model for multiple epochs."""
        if self.optimizer is None:
            self.setup_optimizer()

        train_losses = []
        val_losses = []
        val_accuracies = []
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            # Training
            train_loss = self.train_epoch(train_dataloader)
            train_losses.append(train_loss)

            # Validation
            val_loss, val_accuracy = self.evaluate(val_dataloader)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            # Learning rate scheduling
            self.scheduler.step()

            print(f"Train Loss: {train_loss:.4f}")
            print(
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break

        return train_losses, val_losses, val_accuracies


def create_vocabulary(vocab_size):
    """Create a simple vocabulary mapping."""
    vocab = {'<PAD>': 0}
    for i in range(1, vocab_size):
        vocab[str(i)] = i
    return vocab


def generate_test_sequences(vocab_size, seq_length, num_test_samples, task='copy'):
    """Generate test sequences for evaluation."""
    test_data = []

    for _ in range(num_test_samples):
        input_seq = torch.randint(1, vocab_size, (seq_length,))

        if task == 'copy':
            target_seq = input_seq.clone()
        elif task == 'reverse':
            target_seq = torch.flip(input_seq, dims=[0])
        elif task == 'sort':
            target_seq = torch.sort(input_seq)[0]
        elif task == 'shift':
            target_seq = torch.roll(input_seq, shifts=1, dims=0)

        test_data.append((input_seq, target_seq))

    return test_data


def test_model_performance(model, test_data, vocab, device='cpu'):
    """Test model performance on specific tasks."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for input_seq, target_seq in test_data:
            input_seq = input_seq.unsqueeze(0).to(device)
            target_seq = target_seq.to(device)

            output, _ = model(input_seq)
            predictions = torch.argmax(output, dim=-1)

            # Compare predictions with targets
            if torch.equal(predictions[0], target_seq):
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"Test Accuracy: {accuracy:.4f} ({correct}/{total})")

    return accuracy


def demo_sequence_tasks():
    """Demonstrate the model on different sequence tasks."""
    # Configuration
    vocab_size = 10
    seq_length = 8
    d_model = 64
    num_heads = 4
    num_layers = 2
    batch_size = 32
    num_epochs = 50

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create vocabulary
    vocab = create_vocabulary(vocab_size)

    # Test different tasks
    tasks = ['copy', 'reverse', 'sort', 'shift']

    for task in tasks:
        print(f"\n{'='*50}")
        print(f"Training model for task: {task}")
        print(f"{'='*50}")

        # Create datasets
        train_dataset = SequenceDataset(vocab_size, seq_length, 1000, task)
        val_dataset = SequenceDataset(vocab_size, seq_length, 200, task)

        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)

        # Create model
        model = SimpleTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_length=seq_length
        )

        # Train model
        trainer = Trainer(model, device)
        train_losses, val_losses, val_accuracies = trainer.train(
            train_dataloader, val_dataloader, num_epochs
        )

        # Test on some examples
        test_data = generate_test_sequences(vocab_size, seq_length, 100, task)
        test_model_performance(model, test_data, vocab, device)

        # Show some examples
        print("\nExample predictions:")
        for i in range(3):
            input_seq, target_seq = test_data[i]
            input_seq = input_seq.unsqueeze(0).to(device)

            with torch.no_grad():
                output, attention_weights = model(input_seq)
                predictions = torch.argmax(output, dim=-1)

            input_text = [
                vocab.get(idx.item(), f'<{idx.item()}>') for idx in input_seq[0]]
            target_text = [
                vocab.get(idx.item(), f'<{idx.item()}>') for idx in target_seq]
            pred_text = [
                vocab.get(idx.item(), f'<{idx.item()}>') for idx in predictions[0]]

            print(f"Input:  {' '.join(input_text)}")
            print(f"Target: {' '.join(target_text)}")
            print(f"Pred:   {' '.join(pred_text)}")
            print()


if __name__ == "__main__":
    demo_sequence_tasks()
