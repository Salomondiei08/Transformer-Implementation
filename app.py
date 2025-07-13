from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
from transformer import SimpleTransformer
from trainer import create_vocabulary
import os

app = Flask(__name__)
CORS(app)

# Global variables
model = None
vocab = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model():
    """Load the trained transformer model."""
    global model, vocab

    # Model configuration
    vocab_size = 10
    d_model = 64
    num_heads = 4
    num_layers = 2

    # Create model
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_length=8
    )

    # Load trained weights if available
    model_path = 'model_reverse.pth'  # You can change this to load different models
    try:
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from {model_path}")
        else:
            print("No pre-trained model found. Using untrained model.")
    except Exception as e:
        print(f"Error loading model: {e}. Using untrained model.")

    model.to(device)
    model.eval()

    # Create vocabulary
    vocab = create_vocabulary(vocab_size)
    print("Model loaded successfully!")


def text_to_tokens(text, max_length=8):
    """Convert text input to token sequence."""
    # Simple character-based tokenization for demo
    tokens = []
    for char in text[:max_length]:
        if char.isdigit():
            tokens.append(int(char))
        elif char.isalpha():
            tokens.append(ord(char.lower()) - ord('a') + 1)
        else:
            tokens.append(0)  # Padding for special characters

    # Pad to max_length
    while len(tokens) < max_length:
        tokens.append(0)

    return torch.tensor(tokens[:max_length], dtype=torch.long)


def tokens_to_text(tokens):
    """Convert token sequence back to text."""
    text = ""
    for token in tokens:
        if token == 0:
            continue
        elif 1 <= token <= 9:
            text += str(token.item())
        elif 10 <= token <= 35:
            text += chr(token.item() + ord('a') - 1)
    return text


@app.route('/', methods=['GET'])
def home():
    """Home endpoint."""
    return jsonify({
        "message": "Transformer API is running!",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "tasks": "/tasks"
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Predict endpoint for sequence tasks."""
    try:
        data = request.get_json()
        input_text = data.get('input', '')
        task = data.get('task', 'copy')

        if not input_text:
            return jsonify({"error": "Input text is required"}), 400

        # Convert input to tokens
        input_tokens = text_to_tokens(input_text)
        input_tokens = input_tokens.unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            output, attention_weights = model(input_tokens)
            predictions = torch.argmax(output, dim=-1)

        # Convert prediction back to text
        predicted_text = tokens_to_text(predictions[0])

        # For demo purposes, also provide rule-based predictions
        rule_based_predictions = {
            'copy': input_text,
            'reverse': input_text[::-1],
            'sort': ''.join(sorted(input_text)),
            'shift': ''.join([chr(ord(c) + 1) if c.isalpha() else c for c in input_text])
        }

        return jsonify({
            "input": input_text,
            "task": task,
            "prediction": predicted_text,
            "rule_based_prediction": rule_based_predictions.get(task, predicted_text),
            "attention_weights": attention_weights[0].cpu().numpy().tolist() if attention_weights else []
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/tasks', methods=['GET'])
def get_tasks():
    """Get available tasks."""
    tasks = [
        {
            "id": "copy",
            "name": "Copy",
            "description": "Copy the input sequence exactly"
        },
        {
            "id": "reverse",
            "name": "Reverse",
            "description": "Reverse the input sequence"
        },
        {
            "id": "sort",
            "name": "Sort",
            "description": "Sort characters in ascending order"
        },
        {
            "id": "shift",
            "name": "Shift",
            "description": "Shift each character by +1 position"
        }
    ]
    return jsonify({"tasks": tasks})


if __name__ == '__main__':
    print("Loading transformer model...")
    load_model()
    print("Starting Flask server...")
    port = int(os.environ.get('PORT', 5000))
    print(f"Server will run on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
