# feature_steering.py

import torch
import numpy as np
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from crosscoder import BatchTopKCrosscoder
from data_utils import get_activations, BOS_TOKEN_ID, USER_TOKEN_ID, ASSISTANT_TOKEN_ID
from tqdm import tqdm

def load_crosscoder_model(model_path: str) -> BatchTopKCrosscoder:
    """Load the trained crosscoder model"""
    # Add PosixPath to safe globals for loading
    from pathlib import PosixPath
    import torch.serialization
    torch.serialization.add_safe_globals([PosixPath])
    
    # Load checkpoint
    checkpoint = torch.load(model_path, weights_only=False)
    
    # Extract model parameters
    d_model = checkpoint['W_encoder_ZF'].shape[0] // 2  # Divide by 2 since it's concatenated
    dict_size = checkpoint['W_encoder_ZF'].shape[1]
    k = checkpoint.get('k', 40)  # Default k if not specified
    
    # Create model and load state
    model = BatchTopKCrosscoder(d_model=d_model, dict_size=dict_size, k=k)
    model.load_state_dict(checkpoint)
    
    return model

def get_feature_steering_vector(crosscoder, feature_id, model_type="r1", scale=1.0):
    """
    Extract a steering vector for a specific feature ID.
    
    Args:
        crosscoder: The crosscoder model
        feature_id: The feature ID to extract
        model_type: Either "base" or "r1" to specify which model's decoder to use
        scale: Scaling factor for the steering vector
        
    Returns:
        Steering vector for the specified feature
    """
    # Get decoder weights
    decoder_weights = crosscoder.W_decoder_FZ.data  # Shape: [dict_size, 2*d_model]
    d_model = decoder_weights.shape[1] // 2
    
    # Extract the appropriate decoder weights based on model type
    if model_type == "base":
        decoder = decoder_weights[:, :d_model]  # Shape: [dict_size, d_model]
    elif model_type == "r1":
        decoder = decoder_weights[:, d_model:]  # Shape: [dict_size, d_model]
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Extract the steering vector for the specified feature
    steering_vector = decoder[feature_id]  # Shape: [d_model]
    
    # Scale the steering vector
    steering_vector = steering_vector * scale
    
    return steering_vector

class FeatureSteerer:
    def __init__(self, model, tokenizer, crosscoder, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.crosscoder = crosscoder
        self.device = device
        self.dtype = next(model.parameters()).dtype  # Get model's dtype
        
    def apply_steering_hook(self, layer_num, feature_id, model_type="r1", scale=1.0):
        """
        Apply a hook to inject the steering vector during generation.
        
        Args:
            layer_num: The layer number to apply steering to
            feature_id: The feature ID to steer with
            model_type: Either "base" or "r1" to specify which model's decoder to use
            scale: Scaling factor for the steering vector
        """
        # Get the steering vector and convert to model's dtype
        steering_vector = get_feature_steering_vector(self.crosscoder, feature_id, model_type, scale).to(self.device).to(self.dtype)  # Match model's dtype
        
        # Define the hook function
        def steering_hook(module, input, output):
            # Add the steering vector to the output
            # output shape: [batch_size, seq_len, hidden_size]
            output = output + steering_vector.unsqueeze(0).unsqueeze(0)
            return output
        
        # Get the target layer
        target_layer = self.model.model.layers[layer_num].mlp.down_proj
        
        # Register the hook
        hook_handle = target_layer.register_forward_hook(steering_hook)
        
        return hook_handle
    
    def generate_with_steering(self, prompt, feature_id, layer_num=14, model_type="r1", scale=1.0, max_new_tokens=100, temperature=0.7):
        """
        Generate text with feature steering applied.
        
        Args:
            prompt: The input prompt
            feature_id: The feature ID to steer with
            layer_num: The layer number to apply steering to
            model_type: Either "base" or "r1" to specify which model's decoder to use
            scale: Scaling factor for the steering vector
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for sampling
            
        Returns:
            Generated text with and without steering
        """
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate without steering
        with torch.no_grad():
            outputs_baseline = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True if temperature > 0.0 else False,
                temperature=temperature,
                top_p=0.9,
            )
        
        # Apply steering hook
        hook_handle = self.apply_steering_hook(layer_num, feature_id, model_type, scale)
        
        # Generate with steering
        with torch.no_grad():
            outputs_steered = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True if temperature > 0.0 else False,
                temperature=temperature,
                top_p=0.9,
            )
        
        # Remove the hook
        hook_handle.remove()
        
        # Decode the outputs
        baseline_text = self.tokenizer.decode(outputs_baseline[0], skip_special_tokens=True)
        steered_text = self.tokenizer.decode(outputs_steered[0], skip_special_tokens=True)
        
        return {
            "baseline": baseline_text,
            "steered": steered_text,
            "prompt": prompt,
            "feature_id": feature_id,
            "scale": scale,
            "model_type": model_type
        }
    
    def analyze_feature_impact(self, prompt, feature_id, layer_num=14, scales=[-5.0, -1.0, 1.0, 5.0], model_type="r1", max_new_tokens=100, temperature=0.7):
        """
        Analyze the impact of a feature by applying different scaling factors.
        
        Args:
            prompt: The input prompt
            feature_id: The feature ID to analyze
            layer_num: The layer number to apply steering to
            scales: List of scaling factors to try
            model_type: Either "base" or "r1" to specify which model's decoder to use
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for sampling
            
        Returns:
            Dictionary of results with different scaling factors
        """
        results = {}
        
        # Generate baseline (no steering)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs_baseline = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True if temperature > 0.0 else False,
                temperature=temperature,
                top_p=0.9,
            )
        baseline_text = self.tokenizer.decode(outputs_baseline[0], skip_special_tokens=True)
        results["baseline"] = baseline_text
        
        # Generate with different scaling factors
        for scale in scales:
            hook_handle = self.apply_steering_hook(layer_num, feature_id, model_type, scale)
            
            with torch.no_grad():
                outputs_steered = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True if temperature > 0.0 else False,
                    temperature=temperature,
                    top_p=0.9,
                )
            
            steered_text = self.tokenizer.decode(outputs_steered[0], skip_special_tokens=True)
            results[f"scale_{scale}"] = steered_text
            
            # Remove the hook
            hook_handle.remove()
        
        return results

def run_batch_experiments(feature_ids, prompts, model_types=["r1"], scales=[-5.0, -1.0, 1.0, 5.0], layer_num=14, max_tokens=100, temperature=0.7):
    """
    Run feature steering experiments on multiple prompts and features.
    
    Args:
        feature_ids: List of feature IDs to test
        prompts: List of prompts to test
        model_types: List of model types to use ("base", "r1")
        scales: List of scaling factors to try
        layer_num: Layer number to apply steering to
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling
    """
    # Create output directory
    output_dir = Path("steering_results")
    output_dir.mkdir(exist_ok=True)
    
    # Check for CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load crosscoder model
    print("Loading crosscoder model...")
    crosscoder_path = "crosscoder-layer14_49152_100_fullshuffle_aux.pt"
    crosscoder = load_crosscoder_model(crosscoder_path)
    crosscoder.to(device)
    
    # Process each model type
    for model_type in model_types:
        print(f"\nProcessing model type: {model_type}")
        
        # Load the appropriate model
        if model_type == "base":
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-Math-1.5B", 
                device_map=device, 
                torch_dtype=torch.bfloat16
            )
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
        else:  # r1
            model = AutoModelForCausalLM.from_pretrained(
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 
                device_map=device, 
                torch_dtype=torch.bfloat16
            )
            tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        
        # Create the steerer
        steerer = FeatureSteerer(model, tokenizer, crosscoder, device)
        
        # Process each feature
        for feature_id in feature_ids:
            print(f"\nProcessing feature ID: {feature_id}")
            feature_dir = output_dir / f"feature_{feature_id}_{model_type}"
            feature_dir.mkdir(exist_ok=True)
            
            # Process each prompt
            for i, prompt in enumerate(prompts):
                print(f"  Processing prompt {i+1}/{len(prompts)}")
                
                # Add "Let me think:" to the beginning of the prompt
                thinking_prompt = f"{prompt}"
                
                # Generate results with different scales
                results = steerer.analyze_feature_impact(
                    thinking_prompt, 
                    feature_id,
                    layer_num=layer_num,
                    scales=scales,
                    model_type=model_type,
                    max_new_tokens=max_tokens,
                    temperature=temperature
                )
                
                # Add metadata
                results["feature_id"] = feature_id
                results["model_type"] = model_type
                results["layer_num"] = layer_num
                results["original_prompt"] = prompt
                results["thinking_prompt"] = thinking_prompt
                
                # Save results
                prompt_slug = prompt[:20].replace(" ", "_").replace("?", "").replace(".", "")
                output_file = feature_dir / f"prompt_{i+1}_{prompt_slug}.json"
                
                with open(output_file, "w") as f:
                    import json
                    json.dump(results, f, indent=2)
                
                print(f"    Results saved to {output_file}")
        
        # Free up memory
        del model
        del steerer
        torch.cuda.empty_cache()
    
    print("\nAll experiments completed!")

def main():
    parser = argparse.ArgumentParser(description="Feature steering for language models")
    parser.add_argument("--feature_id", type=int, help="Feature ID to steer with")
    parser.add_argument("--prompt", type=str, help="Input prompt")
    parser.add_argument("--model_type", type=str, default="r1", choices=["base", "r1"], help="Which model's decoder to use")
    parser.add_argument("--scale", type=float, default=1.0, help="Scaling factor for the steering vector")
    parser.add_argument("--layer_num", type=int, default=14, help="Layer number to apply steering to")
    parser.add_argument("--max_tokens", type=int, default=300, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature for sampling") # TODO: Should the temperature be 0.0?
    parser.add_argument("--analyze", action="store_true", help="Analyze feature impact with multiple scales")
    parser.add_argument("--output_file", type=str, help="Output file to save results")
    parser.add_argument("--batch_mode", action="store_true", help="Run batch experiments with predefined prompts")
    args = parser.parse_args()
    
    if args.batch_mode:
        # Define interesting feature IDs to test
        # These are examples - you might want to use specific features you're interested in
        feature_ids = [
            # 17992,   
            # 2385,
            # 11874,
            # 245,
            # 25777,
            # 5542,
            # 33581,
            # 15412, 
            153,
        ]
        
        # Define a set of diverse prompts
        prompts = [
            "Solve this math problem step by step. Put your final answer in \\boxed{}. Problem: Emma had just been given some coins by her parents.  On the way to school she lost exactly half of them, and then by retracing her steps she found exactly four-fifths of the coins she had lost.  What fraction of the coins that she received from her parents were still missing after Emma retraced her steps? Express your answer as a common fraction. Solution: \n<think>\n",
            "Solve this math problem step by step. Put your final answer in \\boxed{}. Problem: Evaluate the infinite geometric series: $$\\frac{1}{3}+\\frac{1}{6}+\\frac{1}{12}+\\frac{1}{24}+\\dots$$ Solution: \n<think>\n",
            "Solve this math problem step by step. Put your final answer in \\boxed{}. Problem: Find the largest prime factor of $9879$. Solution: \n<think>\n"
        ]
        
        # Run batch experiments
        run_batch_experiments(
            feature_ids=feature_ids,
            prompts=prompts,
            model_types=["r1", "base"],  # Test both model types
            scales=[-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0],  # Various scales
            layer_num=14,
            max_tokens=300,
            temperature=0.6
        )
    else:
        # Original functionality
        # Check for CUDA
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Load models
        print("Loading models...")
        crosscoder_path = "crosscoder-layer14_49152_100_fullshuffle_aux.pt"
        crosscoder = load_crosscoder_model(crosscoder_path)
        crosscoder.to(device)
        
        # Load the appropriate model based on model_type
        if args.model_type == "base":
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-Math-1.5B", 
                device_map=device, 
                torch_dtype=torch.bfloat16
            )
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
        else:  # r1
            model = AutoModelForCausalLM.from_pretrained(
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 
                device_map=device, 
                torch_dtype=torch.bfloat16
            )
            tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        
        # Create the steerer
        steerer = FeatureSteerer(model, tokenizer, crosscoder, device)
        
        # Generate with or without analysis
        if args.analyze:
            print(f"Analyzing feature {args.feature_id} with multiple scales...")
            results = steerer.analyze_feature_impact(
                args.prompt, 
                args.feature_id,
                layer_num=args.layer_num,
                scales=[-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0],
                model_type=args.model_type,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature
            )
            
            # Print results
            print(f"\nPrompt: {args.prompt}")
            print(f"\nBaseline output:")
            print(results["baseline"])
            
            for scale in [-5.0, -2.0, -1.0, 1.0, 2.0, 5.0]:
                print(f"\nOutput with scale {scale}:")
                print(results[f"scale_{scale}"])
        else:
            print(f"Generating with feature {args.feature_id}, scale {args.scale}...")
            results = steerer.generate_with_steering(
                args.prompt, 
                args.feature_id,
                layer_num=args.layer_num,
                model_type=args.model_type,
                scale=args.scale,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature
            )
            
            # Print results
            print(f"\nPrompt: {args.prompt}")
            print(f"\nBaseline output:")
            print(results["baseline"])
            print(f"\nSteered output (feature {args.feature_id}, scale {args.scale}):")
            print(results["steered"])
        
        # Save results if output file is specified
        if args.output_file:
            import json
            with open(args.output_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()