import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random


class SequenceDataset(Dataset):
    """Dataset for sequence-to-sequence tasks like copy and reverse."""
    
    def __init__(self, num_samples, seq_len, vocab_size, task='copy', 
                 pad_token=0, start_token=1, end_token=2):
        """
        Args:
            num_samples: Number of samples to generate
            seq_len: Length of sequences (excluding special tokens)
            vocab_size: Size of vocabulary
            task: 'copy' or 'reverse'
            pad_token: Padding token ID
            start_token: Start of sequence token ID
            end_token: End of sequence token ID
        """
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.task = task
        self.pad_token = pad_token
        self.start_token = start_token
        self.end_token = end_token
        
        # Generate vocabulary range (excluding special tokens)
        self.token_range = list(range(3, vocab_size))
        
        self.data = self._generate_data()
    
    def _generate_data(self):
        """Generate all sequences for the dataset."""
        data = []
        
        for _ in range(self.num_samples):
            # Generate random sequence
            seq = [random.choice(self.token_range) for _ in range(self.seq_len)]
            
            # Create input sequence: [START] + seq + [END]
            input_seq = [self.start_token] + seq + [self.end_token]
            
            # Create target sequence based on task
            if self.task == 'copy':
                target_seq = [self.start_token] + seq + [self.end_token]
            elif self.task == 'reverse':
                target_seq = [self.start_token] + seq[::-1] + [self.end_token]
            else:
                raise ValueError(f"Unknown task: {self.task}")
            
            data.append({
                'input': torch.tensor(input_seq, dtype=torch.long),
                'target': torch.tensor(target_seq, dtype=torch.long)
            })
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """Collate function for batching sequences of different lengths."""
    inputs = [item['input'] for item in batch]
    targets = [item['target'] for item in batch]
    
    # Pad sequences to the same length
    inputs_padded = rnn_utils.pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = rnn_utils.pad_sequence(targets, batch_first=True, padding_value=0)
    
    return {
        'input': inputs_padded,
        'target': targets_padded
    }


class VariableLengthSequenceDataset(Dataset):
    """Dataset with variable length sequences for more realistic scenarios."""
    
    def __init__(self, num_samples, min_len=3, max_len=15, vocab_size=20, 
                 task='copy', pad_token=0, start_token=1, end_token=2):
        """
        Args:
            num_samples: Number of samples to generate
            min_len: Minimum sequence length
            max_len: Maximum sequence length
            vocab_size: Size of vocabulary
            task: 'copy' or 'reverse'
            pad_token: Padding token ID
            start_token: Start of sequence token ID
            end_token: End of sequence token ID
        """
        self.num_samples = num_samples
        self.min_len = min_len
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.task = task
        self.pad_token = pad_token
        self.start_token = start_token
        self.end_token = end_token
        
        # Generate vocabulary range (excluding special tokens)
        self.token_range = list(range(3, vocab_size))
        
        self.data = self._generate_data()
    
    def _generate_data(self):
        """Generate all sequences for the dataset."""
        data = []
        
        for _ in range(self.num_samples):
            # Random sequence length
            seq_len = random.randint(self.min_len, self.max_len)
            
            # Generate random sequence
            seq = [random.choice(self.token_range) for _ in range(seq_len)]
            
            # Create input sequence: [START] + seq + [END]
            input_seq = [self.start_token] + seq + [self.end_token]
            
            # Create target sequence based on task
            if self.task == 'copy':
                target_seq = [self.start_token] + seq + [self.end_token]
            elif self.task == 'reverse':
                target_seq = [self.start_token] + seq[::-1] + [self.end_token]
            else:
                raise ValueError(f"Unknown task: {self.task}")
            
            data.append({
                'input': torch.tensor(input_seq, dtype=torch.long),
                'target': torch.tensor(target_seq, dtype=torch.long)
            })
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def create_dataloaders(task='copy', num_samples=5000, seq_len=8, vocab_size=20,
                      batch_size=32, test_split=0.2, variable_length=False):
    """
    Create train and test dataloaders for sequence tasks.
    
    Args:
        task: 'copy' or 'reverse'
        num_samples: Total number of samples
        seq_len: Sequence length (ignored if variable_length=True)
        vocab_size: Vocabulary size
        batch_size: Batch size
        test_split: Fraction of data for testing
        variable_length: Whether to use variable length sequences
    
    Returns:
        train_loader, test_loader
    """
    if variable_length:
        dataset = VariableLengthSequenceDataset(
            num_samples=num_samples,
            min_len=3,
            max_len=seq_len,
            vocab_size=vocab_size,
            task=task
        )
    else:
        dataset = SequenceDataset(
            num_samples=num_samples,
            seq_len=seq_len,
            vocab_size=vocab_size,
            task=task
        )
    
    # Split into train and test
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_loader, test_loader


def tokens_to_string(tokens, vocab_size):
    """Convert token sequence to string representation."""
    token_names = {
        0: '<PAD>',
        1: '<START>',
        2: '<END>'
    }
    
    # Add regular tokens
    for i in range(3, vocab_size):
        token_names[i] = str(i)
    
    return ' '.join([token_names.get(token.item(), f'<UNK{token.item()}>') 
                    for token in tokens])


def print_sample_data(dataloader, vocab_size, num_samples=3):
    """Print some sample data from the dataloader."""
    print("Sample data:")
    print("-" * 50)
    
    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break
        
        inputs = batch['input'][0]  # First sample in batch
        targets = batch['target'][0]
        
        print(f"Sample {i+1}:")
        print(f"Input:  {tokens_to_string(inputs, vocab_size)}")
        print(f"Target: {tokens_to_string(targets, vocab_size)}")
        print()


if __name__ == "__main__":
    # Test the data generation
    print("Testing Copy Task:")
    train_loader, test_loader = create_dataloaders(
        task='copy', 
        num_samples=100,
        seq_len=5,
        vocab_size=10,
        batch_size=4
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print_sample_data(train_loader, vocab_size=10)
    
    print("\n" + "="*60 + "\n")
    
    print("Testing Reverse Task:")
    train_loader_rev, test_loader_rev = create_dataloaders(
        task='reverse',
        num_samples=100,
        seq_len=5,
        vocab_size=10,
        batch_size=4
    )
    
    print_sample_data(train_loader_rev, vocab_size=10)
    
    print("\n" + "="*60 + "\n")
    
    print("Testing Variable Length:")
    train_loader_var, test_loader_var = create_dataloaders(
        task='reverse',
        num_samples=100,
        seq_len=8,  # max length
        vocab_size=10,
        batch_size=4,
        variable_length=True
    )
    
    print_sample_data(train_loader_var, vocab_size=10)