# Task-Specific Transformer Training

This directory contains separate folders for each sequence task, allowing you to train and test different Transformer models independently.

## 📁 Directory Structure

```
tasks/
├── reverse/          # Reverse sequence task
│   └── train_reverse.py
├── copy/            # Copy sequence task  
│   └── train_copy.py
├── sort/            # Sort sequence task
│   └── train_sort.py
├── shift/           # Shift sequence task
│   └── train_shift.py
└── README.md        # This file
```

## 🚀 How to Use

### Option 1: Use the Master Script (Recommended)

```bash
# Run any task from the root directory
python run_all_tasks.py reverse
python run_all_tasks.py copy
python run_all_tasks.py sort
python run_all_tasks.py shift
```

### Option 2: Run Individual Scripts

```bash
# Navigate to specific task folder and run
cd tasks/reverse
python train_reverse.py

cd tasks/copy
python train_copy.py

cd tasks/sort
python train_sort.py

cd tasks/shift
python train_shift.py
```

## 📊 Task Descriptions

### 🔄 Reverse Task

- **Input**: `[1, 2, 3, 4, 5, 6, 7, 8]`
- **Output**: `[8, 7, 6, 5, 4, 3, 2, 1]`
- **Difficulty**: Medium
- **Model**: `model_reverse.pth`

### 📋 Copy Task

- **Input**: `[1, 2, 3, 4, 5, 6, 7, 8]`
- **Output**: `[1, 2, 3, 4, 5, 6, 7, 8]`
- **Difficulty**: Easy
- **Model**: `model_copy.pth`

### 🔢 Sort Task

- **Input**: `[8, 3, 1, 6, 4, 7, 2, 5]`
- **Output**: `[1, 2, 3, 4, 5, 6, 7, 8]`
- **Difficulty**: Hard
- **Model**: `model_sort.pth`

### ➡️ Shift Task

- **Input**: `[1, 2, 3, 4, 5, 6, 7, 8]`
- **Output**: `[8, 1, 2, 3, 4, 5, 6, 7]`
- **Difficulty**: Hard
- **Model**: `model_shift.pth`

## 🎯 Interactive Testing

Each training script includes an interactive mode where you can:

1. Enter your own 8-number sequences
2. See the model's predictions
3. Test the model's performance

Example:

```
Enter sequence: 1 2 3 4 5 6 7 8
Predicted reverse: ['8', '7', '6', '5', '4', '3', '2', '1']
```

## 📈 Expected Performance

- **Copy**: Usually achieves 100% accuracy quickly
- **Reverse**: Achieves high accuracy (95%+) within 20-30 epochs
- **Sort**: May take longer to converge, accuracy varies
- **Shift**: Most challenging, may require more epochs

## 🔧 Configuration

Each task uses the same configuration:

- Vocabulary size: 10 (numbers 1-9)
- Sequence length: 8
- Model dimension: 64
- Attention heads: 4
- Layers: 2
- Training epochs: 30

You can modify these parameters in each individual training script.

## 📁 Output Files

Each task folder will contain:

- `model_[task].pth` - Trained model weights
- Training logs and example predictions

## 🎓 Learning Insights

- **Copy**: Tests basic sequence memorization
- **Reverse**: Tests positional understanding and attention patterns
- **Sort**: Tests numerical reasoning and comparison
- **Shift**: Tests temporal relationships and circular patterns

Each task demonstrates different aspects of how Transformers learn sequence transformations!
