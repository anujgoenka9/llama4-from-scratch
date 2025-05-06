import torch
import pickle
import numpy as np
import math
import argparse
from torch.nn import functional as F
from Llama4 import Llama4ForCausalLM, BPETokenizer

def load_model_and_tokenizer(model_path, tokenizer_path, device):
    """Load the saved model and tokenizer"""
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract config from checkpoint
    config = checkpoint['config']
    
    # Initialize model
    model = Llama4ForCausalLM(config).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    
    # Load tokenizer
    print(f"Loading tokenizer from {tokenizer_path}...")
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    return model, tokenizer, config

def calculate_perplexity(model, tokenizer, text, device, max_length=512):
    """Calculate perplexity on a given text"""
    # Tokenize text
    tokens = tokenizer.encode(text)
    
    # Handle longer texts by splitting into chunks
    if len(tokens) <= max_length:
        chunks = [tokens]
    else:
        # Create overlapping chunks with context
        chunks = []
        for i in range(0, len(tokens), max_length // 2):
            chunk = tokens[i:i + max_length]
            chunks.append(chunk)
    
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for chunk in chunks:
            if len(chunk) < 2:  # Need at least 2 tokens (input and target)
                continue
                
            # Prepare input and target
            inputs = chunk[:-1]
            targets = chunk[1:]
            
            # Convert to tensors
            input_ids = torch.tensor([inputs]).to(device)
            target_ids = torch.tensor([targets]).to(device)
            
            # Forward pass
            logits, _, _ = model(input_ids)
            
            # Compute loss
            # Reshape logits to [batch_size * seq_length, vocab_size]
            shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            # Reshape targets to [batch_size * seq_length]
            shift_targets = target_ids.contiguous().view(-1)
            
            # Calculate cross-entropy loss
            loss = F.cross_entropy(shift_logits, shift_targets, reduction='sum')
            
            total_loss += loss.item()
            total_tokens += len(targets)
    
    # Calculate perplexity
    perplexity = math.exp(total_loss / total_tokens)
    return perplexity

def evaluate_generation_quality(model, tokenizer, prompts, device, generation_params):
    """Evaluate the quality of generated text using multiple metrics"""
    results = []
    
    for prompt in prompts:
        # Convert prompt to tensor
        input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
        
        # Generate text
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                max_length=generation_params['max_length'],
                temperature=generation_params['temperature'],
                top_k=generation_params['top_k'],
                top_p=generation_params['top_p'],
                repetition_penalty=generation_params['repetition_penalty'],
                do_sample=generation_params['do_sample']
            )
        
        # Decode generated ids
        generated_text = tokenizer.decode(generated_ids[0].tolist())
        
        # Calculate token diversity (unique tokens / total tokens)
        generated_tokens = tokenizer.encode(generated_text)
        token_diversity = len(set(generated_tokens)) / len(generated_tokens) if generated_tokens else 0
        
        # Calculate repetition rate (repeated n-grams / total n-grams)
        def count_repeated_ngrams(text, n=3):
            tokens = tokenizer.encode(text)
            ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
            if not ngrams:
                return 0
            ngram_counts = {}
            for ngram in ngrams:
                if ngram in ngram_counts:
                    ngram_counts[ngram] += 1
                else:
                    ngram_counts[ngram] = 1
            repeated_ngrams = sum(1 for count in ngram_counts.values() if count > 1)
            repetition_rate = repeated_ngrams / len(ngrams) if ngrams else 0
            return repetition_rate
        
        repetition_rate = count_repeated_ngrams(generated_text)
        
        results.append({
            'prompt': prompt,
            'generated_text': generated_text,
            'token_diversity': token_diversity,
            'repetition_rate': repetition_rate
        })
    
    return results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate the trained Llama4 model')
    parser.add_argument('--model_path', type=str, default='llama4_enhanced_model.pt', help='Path to the saved model')
    parser.add_argument('--tokenizer_path', type=str, default='llama4_enhanced_tokenizer.pkl', help='Path to the saved tokenizer')
    parser.add_argument('--test_file', type=str, help='Path to test file with evaluation text')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer, config = load_model_and_tokenizer(args.model_path, args.tokenizer_path, args.device)
    
    # Print model info
    print(f"Model size: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
    print(f"Model config: {config['num_layers']} layers, {config['hidden_size']} hidden size")
    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    
    # Prompts for generation evaluation
    prompts = [
        "CRISPR technology is used for",
        "The main advantage of gene editing with CRISPR is",
        "Scientists use Cas9 protein to",
        "The ethical concerns about CRISPR include"
    ]
    
    # Generation parameters
    generation_params = {
        'max_length': 100,
        'temperature': 0.8,
        'top_k': 50,
        'top_p': 0.9,
        'repetition_penalty': 1.2,
        'do_sample': True
    }
    
    # Evaluate generation quality
    print("\n=== Evaluating Generation Quality ===")
    generation_results = evaluate_generation_quality(model, tokenizer, prompts, args.device, generation_params)
    
    for i, result in enumerate(generation_results):
        print(f"\nPrompt {i+1}: {result['prompt']}")
        print(f"Generated Text: {result['generated_text']}")
        print(f"Token Diversity: {result['token_diversity']:.4f}")
        print(f"Repetition Rate: {result['repetition_rate']:.4f}")
    
    # Calculate average metrics
    avg_diversity = np.mean([r['token_diversity'] for r in generation_results])
    avg_repetition = np.mean([r['repetition_rate'] for r in generation_results])
    print(f"\nAverage Token Diversity: {avg_diversity:.4f}")
    print(f"Average Repetition Rate: {avg_repetition:.4f}")
    
    # Calculate perplexity if test file is provided
    if args.test_file:
        print("\n=== Calculating Perplexity ===")
        with open(args.test_file, 'r') as f:
            test_text = f.read()
        
        perplexity = calculate_perplexity(model, tokenizer, test_text, args.device)
        print(f"Perplexity: {perplexity:.4f}")

if __name__ == "__main__":
    main() 