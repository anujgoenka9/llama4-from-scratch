# Llama 4 From Scratch

This project implements the Llama 4 architecture from scratch using PyTorch, including key components like Mixture of Experts (MoE), Rotary Positional Embeddings (RoPE), and Grouped-Query Attention (GQA). The implementation supports training models from small CPU-friendly sizes up to billion-parameter scale on GPUs.

## Features

The implementation includes:

- **BPE Tokenizer**: Custom implementation of Byte-Pair Encoding for tokenization
- **Mixture-of-Experts (MoE)**: Sparse router-based architecture with top-k routing
- **Rotary Positional Embeddings (RoPE)**: For handling positional information
- **Grouped-Query Attention (GQA)**: For efficient attention computation
- **RMSNorm**: Used instead of traditional LayerNorm for better performance
- **SwiGLU Activations**: In the feed-forward networks
- **Key-Value Caching**: For efficient auto-regressive text generation
- **Mixed Precision Training**: For faster training on GPUs

## Model Configurations

The code supports two main configurations:

### CPU-Friendly Model (24M parameters)
- 6 transformer decoder layers
- 384 hidden dimension size
- 12 attention heads
- 4 key-value heads (for GQA)
- MoE layers at positions 1, 3, 5

### GPU 1B Parameter Model
- 24 transformer decoder layers
- 2048 hidden dimension size
- 32 attention heads
- 8 key-value heads (for GQA)
- MoE layers at every third layer
- 16 experts per MoE layer with 2 active experts per token

## Installation

1. Clone the repository:
```
git clone https://github.com/anujgoenka06/llama4-from-scratch.git
cd llama4-from-scratch
```

2. Create a virtual environment:
```
python -m venv llama4_venv
source llama4_venv/bin/activate  # On Windows, use: llama4_venv\Scripts\activate
```

3. Install dependencies:
```
pip install torch numpy
```

## Usage

### Adding Training Data

The model trains on a corpus of text about CRISPR gene editing. You can add your own training data:

1. Edit `additional_corpus.txt` to add more paragraphs
2. Make sure each paragraph is separated by a blank line
3. The model will automatically load this data during training

### Training

#### CPU Training (Small Model)

For training on a CPU:

```
python Llama4.py
```

The script automatically detects if you're on CPU and uses the smaller configuration.

#### GPU Training (1B Parameter Model)

If you have a GPU with sufficient memory (16GB+ recommended):

```
python Llama4.py
```

When a CUDA device is detected, the script will use the 1B parameter configuration.

#### Training Features

- **Automatic checkpointing**: The model saves checkpoints every 5 epochs and whenever a new best loss is achieved
- **Resume training**: Training will automatically resume from the latest checkpoint if available
- **Early stopping**: Training stops after a specified number of epochs without improvement
- **Mixed precision**: Automatically uses FP16 precision when on GPU for faster training

### Inference

Generate text using a trained model:

```
python inference.py --prompt "CRISPR technology" --temperature 0.8 --top_p 0.9
```

For interactive mode with multiple prompts:

```
python inference.py --interactive
```

#### Inference Parameters

- `--model_path`: Path to the saved model (default: llama4_enhanced_model.pt)
- `--tokenizer_path`: Path to the saved tokenizer (default: llama4_enhanced_tokenizer.pkl)
- `--prompt`: Starting prompt for generation
- `--temperature`: Controls randomness (higher = more random)
- `--top_k`: Limits sampling to top k tokens
- `--top_p`: Nucleus sampling probability threshold
- `--repetition_penalty`: Reduces repetition of tokens
- `--do_sample`: Whether to use sampling (default: True)

### Evaluation

Evaluate model quality with perplexity and other metrics:

```
python evaluate.py --test_file test_text.txt
```

The evaluation script:
- Tests the model on predefined prompts about CRISPR
- Calculates token diversity and repetition metrics
- Computes perplexity on test text if provided

## Performance Considerations

### CPU Training

- The CPU model is designed to train in a reasonable amount of time on modern CPUs
- Expect training to take several hours
- Consider reducing context length or batch size if memory issues occur

### GPU Training

- The 1B parameter model requires a high-end GPU with 16GB+ VRAM
- Using mixed precision reduces memory requirements
- If you encounter CUDA out-of-memory errors:
  1. Reduce batch size (by editing `Llama4.py`)
  2. Increase gradient accumulation steps
  3. Reduce model size by decreasing layers or hidden dimensions

## Advanced Features

For researchers who want to modify the implementation:

- The architecture is modular with separate classes for each component
- Key hyperparameters can be adjusted in the config dictionary
- Custom data can be added via the `additional_corpus.txt` file
- Checkpoint frequency and early stopping patience are configurable

## Files

- `Llama4.py`: The main implementation with model architecture and training code
- `inference.py`: Script for text generation using a trained model
- `evaluate.py`: Script for evaluating model performance
- `test_text.txt`: Example text for perplexity evaluation
- `additional_corpus.txt`: File for adding more training data