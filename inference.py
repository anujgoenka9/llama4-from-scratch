import torch
import pickle
import argparse
from Llama4 import Llama4ForCausalLM, BPETokenizer

def load_model_and_tokenizer(model_path, tokenizer_path):
    """Load the saved model and tokenizer"""
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extract config from checkpoint
    config = checkpoint['config']
    
    # Initialize model
    model = Llama4ForCausalLM(config)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load tokenizer
    print(f"Loading tokenizer from {tokenizer_path}...")
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    return model, tokenizer, config

def generate_text(model, tokenizer, prompt, params):
    """Generate text using the model"""
    # Convert prompt to tensor
    input_ids = torch.tensor([tokenizer.encode(prompt)]).to(params['device'])
    prompt_length = len(input_ids[0])
    
    # Move model to device
    model = model.to(params['device'])
    model.eval()
    
    # Make sure we generate at least some new tokens
    min_new_tokens = 30
    max_length = max(prompt_length + min_new_tokens, params['max_length'])
    
    # Generate text
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_length=max_length,
            temperature=params['temperature'],
            top_k=params['top_k'],
            top_p=params['top_p'],
            repetition_penalty=params['repetition_penalty'],
            do_sample=params['do_sample']
        )
    
    # Decode generated ids
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    
    return generated_text

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate text using the trained Llama4 model')
    parser.add_argument('--model_path', type=str, default='llama4_enhanced_model.pt', help='Path to the saved model')
    parser.add_argument('--tokenizer_path', type=str, default='llama4_enhanced_tokenizer.pkl', help='Path to the saved tokenizer')
    parser.add_argument('--prompt', type=str, default='CRISPR technology', help='Prompt for text generation')
    parser.add_argument('--max_length', type=int, default=200, help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=0.9, help='Temperature for sampling (higher = more random)')
    parser.add_argument('--top_k', type=int, default=40, help='Top-k sampling parameter')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling parameter')
    parser.add_argument('--repetition_penalty', type=float, default=1.2, help='Penalty for repeating tokens')
    parser.add_argument('--do_sample', action='store_true', default=True, help='Whether to use sampling (default: True)')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer, config = load_model_and_tokenizer(args.model_path, args.tokenizer_path)
    
    # Print model info
    print(f"Model size: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
    print(f"Model config: {config['num_layers']} layers, {config['hidden_size']} hidden size")
    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    
    # Generate text
    generation_params = {
        'device': args.device,
        'max_length': args.max_length,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'repetition_penalty': args.repetition_penalty,
        'do_sample': args.do_sample
    }
    
    print(f"\nGeneration parameters: temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
    
    if args.interactive:
        print("\n=== Interactive Mode ===")
        print("Enter prompts to generate text. Type 'exit' to quit.")
        
        while True:
            prompt = input("\nPrompt> ")
            if prompt.lower() == 'exit':
                break
                
            print("Generating...")
            generated_text = generate_text(model, tokenizer, prompt, generation_params)
            print(f"\nGenerated Text:\n{generated_text}")
    else:
        print(f"\nPrompt: '{args.prompt}'")
        print("Generating...")
        generated_text = generate_text(model, tokenizer, args.prompt, generation_params)
        print(f"\nGenerated Text:\n{generated_text}")

if __name__ == "__main__":
    main() 