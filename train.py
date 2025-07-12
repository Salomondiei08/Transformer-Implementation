import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
import json
from datetime import datetime

from transformer import MiniTransformer, count_parameters
from data import create_dataloaders, tokens_to_string


class Trainer:
    """Trainer class for the Mini Transformer."""
    
    def __init__(self, model, device='cpu', learning_rate=1e-3, 
                 weight_decay=1e-4, clip_grad=1.0):
        self.model = model.to(device)
        self.device = device
        self.clip_grad = clip_grad
        
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    def compute_accuracy(self, outputs, targets):
        """Compute sequence accuracy (exact match)."""
        # Get predictions
        preds = torch.argmax(outputs, dim=-1)
        
        # Create mask to ignore padding tokens
        mask = (targets != 0)
        
        # Check if entire sequences match (ignoring padding)
        correct_sequences = []
        for i in range(preds.size(0)):
            pred_seq = preds[i][mask[i]]
            target_seq = targets[i][mask[i]]
            correct_sequences.append(torch.equal(pred_seq, target_seq))
        
        accuracy = sum(correct_sequences) / len(correct_sequences)
        return accuracy
    
    def compute_token_accuracy(self, outputs, targets):
        """Compute token-level accuracy."""
        preds = torch.argmax(outputs, dim=-1)
        mask = (targets != 0)  # Ignore padding tokens
        
        correct_tokens = (preds == targets) & mask
        total_tokens = mask.sum()
        
        if total_tokens == 0:
            return 0.0
        
        token_accuracy = correct_tokens.sum().float() / total_tokens.float()
        return token_accuracy.item()
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_token_accuracy = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # Forward pass
            outputs = self.model(inputs, use_look_ahead_mask=True)
            
            # Compute loss
            loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.clip_grad > 0:
                clip_grad_norm_(self.model.parameters(), self.clip_grad)
            
            self.optimizer.step()
            
            # Compute metrics
            token_accuracy = self.compute_token_accuracy(outputs, targets)
            
            total_loss += loss.item()
            total_token_accuracy += token_accuracy
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'token_acc': f'{token_accuracy:.4f}'
            })
        
        avg_loss = total_loss / num_batches
        avg_token_accuracy = total_token_accuracy / num_batches
        
        return avg_loss, avg_token_accuracy
    
    def evaluate(self, val_loader):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        total_token_accuracy = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # Forward pass
                outputs = self.model(inputs, use_look_ahead_mask=True)
                
                # Compute loss
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                
                # Compute accuracies
                accuracy = self.compute_accuracy(outputs, targets)
                token_accuracy = self.compute_token_accuracy(outputs, targets)
                
                total_loss += loss.item()
                total_accuracy += accuracy
                total_token_accuracy += token_accuracy
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_token_accuracy = total_token_accuracy / num_batches
        
        return avg_loss, avg_accuracy, avg_token_accuracy
    
    def generate_sample(self, input_seq, max_length=50, vocab_size=20):
        """Generate a sequence autoregressively."""
        self.model.eval()
        
        with torch.no_grad():
            # Start with input sequence
            generated = input_seq.clone()
            
            for _ in range(max_length):
                # Forward pass
                outputs = self.model(generated, use_look_ahead_mask=True)
                
                # Get next token prediction
                next_token_logits = outputs[0, -1, :]  # Last token of first sequence
                next_token = torch.argmax(next_token_logits)
                
                # Append to sequence
                generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                
                # Stop if we generate end token
                if next_token.item() == 2:  # End token
                    break
            
            return generated
    
    def print_sample_predictions(self, val_loader, vocab_size, num_samples=3):
        """Print some sample predictions."""
        self.model.eval()
        
        print("\nSample Predictions:")
        print("-" * 60)
        
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= num_samples:
                    break
                
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # Forward pass
                outputs = self.model(inputs, use_look_ahead_mask=True)
                predictions = torch.argmax(outputs, dim=-1)
                
                # Show first sample in batch
                input_seq = inputs[0]
                target_seq = targets[0]
                pred_seq = predictions[0]
                
                print(f"Sample {i+1}:")
                print(f"Input:  {tokens_to_string(input_seq, vocab_size)}")
                print(f"Target: {tokens_to_string(target_seq, vocab_size)}")
                print(f"Pred:   {tokens_to_string(pred_seq, vocab_size)}")
                
                # Check if prediction matches target
                mask = (target_seq != 0)
                correct = torch.equal(pred_seq[mask], target_seq[mask])
                print(f"Correct: {correct}")
                print()
    
    def train(self, train_loader, val_loader, epochs, vocab_size, 
              save_path='models', print_samples=True):
        """Main training loop."""
        best_val_accuracy = 0
        
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Model has {count_parameters(self.model):,} parameters")
        print("-" * 60)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss, train_token_acc = self.train_epoch(train_loader)
            
            # Evaluate
            val_loss, val_accuracy, val_token_acc = self.evaluate(val_loader)
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            
            # Print results
            print(f"Train Loss: {train_loss:.4f}, Train Token Acc: {train_token_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val Token Acc: {val_token_acc:.4f}")
            
            # Print sample predictions
            if print_samples and (epoch + 1) % 5 == 0:
                self.print_sample_predictions(val_loader, vocab_size)
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.save_model(os.path.join(save_path, 'best_model.pt'))
                print(f"New best model saved! Accuracy: {val_accuracy:.4f}")
        
        print(f"\nTraining completed! Best validation accuracy: {best_val_accuracy:.4f}")
        
        # Save final model and training history
        self.save_model(os.path.join(save_path, 'final_model.pt'))
        self.save_training_history(os.path.join(save_path, 'training_history.json'))
        
        return best_val_accuracy
    
    def save_model(self, path):
        """Save model state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
        }, path)
    
    def load_model(self, path):
        """Load model state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
    
    def save_training_history(self, path):
        """Save training history as JSON."""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(path, 'w') as f:
            json.dump(history, f, indent=2)
    
    def plot_training_history(self, save_path=None):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot losses
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, label='Train Loss', marker='o')
        ax1.plot(epochs, self.val_losses, label='Val Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(epochs, self.val_accuracies, label='Val Accuracy', marker='s', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def main():
    """Main training function."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    vocab_size = 20
    d_model = 128
    num_heads = 8
    num_layers = 4
    d_ff = 512
    seq_len = 8
    batch_size = 32
    learning_rate = 1e-3
    epochs = 50
    
    # Create model
    model = MiniTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_len=100
    )
    
    # Train on copy task
    print("Training on COPY task:")
    print("=" * 60)
    
    train_loader, val_loader = create_dataloaders(
        task='copy',
        num_samples=5000,
        seq_len=seq_len,
        vocab_size=vocab_size,
        batch_size=batch_size
    )
    
    trainer = Trainer(model, device=device, learning_rate=learning_rate)
    copy_accuracy = trainer.train(
        train_loader, val_loader, epochs, vocab_size, 
        save_path='models/copy_task'
    )
    
    # Plot training history
    trainer.plot_training_history('models/copy_task/training_plot.png')
    
    # Train on reverse task (create new model)
    print("\n\nTraining on REVERSE task:")
    print("=" * 60)
    
    model_reverse = MiniTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_len=100
    )
    
    train_loader_rev, val_loader_rev = create_dataloaders(
        task='reverse',
        num_samples=5000,
        seq_len=seq_len,
        vocab_size=vocab_size,
        batch_size=batch_size
    )
    
    trainer_reverse = Trainer(model_reverse, device=device, learning_rate=learning_rate)
    reverse_accuracy = trainer_reverse.train(
        train_loader_rev, val_loader_rev, epochs, vocab_size,
        save_path='models/reverse_task'
    )
    
    # Plot training history
    trainer_reverse.plot_training_history('models/reverse_task/training_plot.png')
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY:")
    print(f"Copy Task - Best Accuracy: {copy_accuracy:.4f}")
    print(f"Reverse Task - Best Accuracy: {reverse_accuracy:.4f}")


if __name__ == "__main__":
    main()