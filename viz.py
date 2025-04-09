import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
from typing import Dict, List, Tuple, Optional, Union
import os
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from crosscoder import BatchTopKCrosscoder
from data_utils import get_activations, BOS_TOKEN_ID, USER_TOKEN_ID, ASSISTANT_TOKEN_ID

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

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
    
    # Ensure model is in float32
    model = model.to(torch.float32)
    
    return model

def analyze_decoder_norm_diffs(decoder_A: torch.Tensor, decoder_B: torch.Tensor) -> np.ndarray:
    """
    Analyze decoder norm differences to understand feature exclusivity.
    Returns values between -1 (base model only) and 1 (R1 model only).
    """
    # Calculate norms across the d_model dimension
    norm_A = torch.norm(decoder_A, dim=1)  # Shape: [dict_size]
    norm_B = torch.norm(decoder_B, dim=1)  # Shape: [dict_size]
    
    # Calculate normalized difference
    norm_diff = (norm_B - norm_A) / (norm_B + norm_A + 1e-10)
    
    return norm_diff.cpu().numpy()

def get_crosscoder_activations(prompt: str, response: str = "", layer_num: int = 14, ctx_len=1024):
    """Get crosscoder activations for a given prompt"""
    # Tokenize the prompt
    prompt_tokens = r1_tokenizer(prompt, add_special_tokens=False)["input_ids"]
    
    if response:
        response_tokens = r1_tokenizer(response)["input_ids"][1:]  # remove BOS from response
        # Add special tokens
        tokens = [BOS_TOKEN_ID] + [USER_TOKEN_ID] + prompt_tokens + [ASSISTANT_TOKEN_ID] + response_tokens
    else:
        # Just the prompt without response
        tokens = [BOS_TOKEN_ID] + [USER_TOKEN_ID] + prompt_tokens
    
    tokens = tokens[:ctx_len]
    tokens = torch.tensor(tokens).unsqueeze(0).to(device)
    
    # Get activations from both models
    with torch.no_grad():
        base_activations = get_activations(base_model, tokens, layer_num)
        r1_activations = get_activations(r1_model, tokens, layer_num)
    
    # Convert to float32 before concatenation to match crosscoder's expected type
    base_activations = base_activations.to(torch.float32)
    r1_activations = r1_activations.to(torch.float32)
    
    # Concatenate activations
    concatenated_activations = torch.cat([base_activations, r1_activations], dim=-1)
    
    # Reshape to match crosscoder input format
    batch_size, seq_len, hidden_size = concatenated_activations.shape
    reshaped_activations = concatenated_activations.reshape(-1, hidden_size)
    
    # Get crosscoder activations
    with torch.no_grad():
        output = crosscoder(reshaped_activations)
        features = output["sparse_activations"].reshape(batch_size, seq_len, -1)
    
    return features, tokens

def find_top_activating_tokens(feature_id: int, num_examples: int = 100, limit: int = 10):
    """Find tokens that most strongly activate a specific feature"""
    # Convert feature_id to integer
    feature_id = int(feature_id)
    
    # Load the dataset
    dataset = load_dataset("ServiceNow-AI/R1-Distill-SFT", "v1", split="train", streaming=False)
    dataset = dataset.shuffle(seed=42)
    print(f"Loaded {len(dataset)} examples from dataset")
    
    top_activations = []
    
    for i, example in enumerate(tqdm(dataset, total=min(limit, len(dataset)))):
        if i >= limit:
            break
        
        prompt = example["reannotated_messages"][0]["content"]
        response = example["reannotated_messages"][1]["content"]
        
        features, tokens = get_crosscoder_activations(prompt, response)
        
        # Get activations for the specified feature
        feature_acts = features[0, :, feature_id].cpu().numpy()
        
        # Find top activating positions
        top_indices = np.argsort(feature_acts)[-5:][::-1]  # Get top 5 positions per example
        
        for idx in top_indices:
            if feature_acts[idx] > 0:  # Only include positive activations
                token = r1_tokenizer.decode(tokens[0, idx])
                
                # Get context around the token (5 tokens before and after)
                start_idx = max(0, idx - 5)
                end_idx = min(tokens.shape[1], idx + 6)
                context_tokens = tokens[0, start_idx:end_idx].cpu().tolist()
                context = r1_tokenizer.decode(context_tokens)
                
                # Get the full text with highlighted token
                full_text = prompt + "\n\n" + response
                token_positions = []
                
                # Create token-level activations for highlighting
                all_tokens = r1_tokenizer.encode(full_text)
                token_activations = []
                
                # Process the full text to get activations for all tokens
                full_features, full_tokens = get_crosscoder_activations(prompt, response)
                full_acts = full_features[0, :, feature_id].cpu().numpy()
                
                top_activations.append({
                    'token': token,
                    'activation': float(feature_acts[idx]),
                    'context': context,
                    'full_text': full_text,
                    'prompt': prompt,
                    'response': response,
                    'token_index': int(idx),
                    'all_activations': full_acts.tolist()
                })
    
    # Sort by activation strength
    top_activations = sorted(top_activations, key=lambda x: x['activation'], reverse=True)
    
    return top_activations[:num_examples]

def generate_html_file():
    """Generate an interactive HTML file for exploring feature activations"""
    # Load the crosscoder model
    model_path = "crosscoder-layer14_49152_100_fullshuffle_aux.pt"
    global crosscoder, base_model, r1_model, r1_tokenizer
    
    # Load models
    print("Loading models...")
    crosscoder = load_crosscoder_model(model_path)
    crosscoder.to(device)
    
    base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B", device_map=device, torch_dtype=torch.bfloat16)
    r1_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    r1_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", device_map=device, torch_dtype=torch.bfloat16)
    
    # Get decoder weights
    decoder_weights = crosscoder.W_decoder_FZ.data
    d_model = decoder_weights.shape[1] // 2
    
    # Split decoder weights for base and R1 models
    decoder_A = decoder_weights[:, :d_model]
    decoder_B = decoder_weights[:, d_model:]
    
    # Calculate decoder norm differences
    norm_diffs = analyze_decoder_norm_diffs(decoder_A, decoder_B)
    
    # Create a DataFrame for easier analysis
    df = pd.DataFrame({'feature_id': np.arange(len(norm_diffs)), 'dec_norm_diff': norm_diffs})
    
    # Sort features by norm difference
    df_sorted = df.sort_values(by='dec_norm_diff')
    
    # Generate distribution plot
    plt.figure(figsize=(10, 6))
    plt.hist(norm_diffs, bins=50)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Distribution of Decoder Norm Differences')
    plt.xlabel('Norm Difference (negative = base, positive = R1)')
    plt.ylabel('Count')
    
    # Save plot to base64 for embedding in HTML
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.read()).decode('utf-8')
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Feature Explorer</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                line-height: 1.6;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            h1, h2, h3 {{
                color: #333;
            }}
            .feature-info {{
                background-color: #f5f5f5;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .search-box {{
                margin: 20px 0;
                padding: 15px;
                background-color: #eef;
                border-radius: 5px;
            }}
            input[type="number"] {{
                padding: 8px;
                width: 100px;
            }}
            button {{
                padding: 8px 15px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }}
            button:hover {{
                background-color: #45a049;
            }}
            .example {{
                margin-bottom: 30px;
                padding: 15px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            .token {{
                display: inline-block;
                padding: 2px 5px;
                margin: 2px;
                border-radius: 3px;
            }}
            .highlighted {{
                background-color: rgba(255, 0, 0, 0.3);
                font-weight: bold;
            }}
            .activation-bar {{
                height: 5px;
                background-color: #4CAF50;
                margin-top: 5px;
            }}
            .results {{
                display: none;
            }}
            .loading {{
                display: none;
                text-align: center;
                margin: 20px 0;
            }}
            .distribution-plot {{
                text-align: center;
                margin: 20px 0;
            }}
            pre {{
                white-space: pre-wrap;
                word-wrap: break-word;
                background-color: #f8f8f8;
                padding: 10px;
                border-radius: 5px;
                overflow-x: auto;
            }}
            .token-highlight {{
                background-color: rgba(255, 0, 0, 0.3);
            }}
        </style>
        <script>
            // Store norm differences for all features
            const normDiffs = {json.dumps(norm_diffs.tolist())};
            
            // Function to search for a feature
            async function searchFeature() {{
                // Show loading indicator
                document.getElementById('loading').style.display = 'block';
                document.getElementById('results').style.display = 'none';
                
                const featureId = parseInt(document.getElementById('featureId').value);
                if (isNaN(featureId) || featureId < 0 || featureId >= normDiffs.length) {{
                    alert('Please enter a valid feature ID');
                    document.getElementById('loading').style.display = 'none';
                    return;
                }}
                
                // Display feature info
                const normDiff = normDiffs[featureId];
                let featureType = "Shared";
                if (normDiff > 0.95) {{
                    featureType = "R1-exclusive";
                }} else if (normDiff < -0.95) {{
                    featureType = "Base-exclusive";
                }}
                
                document.getElementById('featureInfo').innerHTML = `
                    <h3>Feature #${{featureId}}</h3>
                    <p><strong>Decoder Norm Difference:</strong> ${{normDiff.toFixed(4)}}</p>
                    <p><strong>Feature Type:</strong> ${{featureType}}</p>
                `;
                
                try {{
                    // Fetch top activations from server
                    const response = await fetch('/get_activations?feature_id=' + featureId);
                    const data = await response.json();
                    
                    // Display results
                    const resultsContainer = document.getElementById('activationResults');
                    resultsContainer.innerHTML = '';
                    
                    data.forEach((example, index) => {{
                        const div = document.createElement('div');
                        div.className = 'example';
                        
                        // Create header with activation info
                        const header = document.createElement('h3');
                        header.textContent = `Example #${{index + 1}} - Token: "${{example.token}}" (Activation: ${{example.activation.toFixed(4)}})`;
                        div.appendChild(header);
                        
                        // Create prompt section
                        const promptHeader = document.createElement('h4');
                        promptHeader.textContent = 'Prompt:';
                        div.appendChild(promptHeader);
                        
                        const promptPre = document.createElement('pre');
                        promptPre.textContent = example.prompt;
                        div.appendChild(promptPre);
                        
                        // Create response section with highlighted tokens
                        const responseHeader = document.createElement('h4');
                        responseHeader.textContent = 'Response:';
                        div.appendChild(responseHeader);
                        
                        const responsePre = document.createElement('pre');
                        
                        // Tokenize the response and highlight tokens with high activation
                        const tokens = example.response.split('');
                        const activations = example.all_activations;
                        
                        // Find the activation threshold (use 0.5 of max activation as threshold)
                        const maxActivation = Math.max(...activations.filter(a => a > 0));
                        const threshold = maxActivation * 0.5;
                        
                        // Create highlighted HTML
                        let highlightedText = '';
                        let currentHighlight = false;
                        
                        for (let i = 0; i < tokens.length; i++) {{
                            const activation = i < activations.length ? activations[i] : 0;
                            const shouldHighlight = activation > threshold;
                            
                            if (shouldHighlight && !currentHighlight) {{
                                highlightedText += '<span class="token-highlight">';
                                currentHighlight = true;
                            }} else if (!shouldHighlight && currentHighlight) {{
                                highlightedText += '</span>';
                                currentHighlight = false;
                            }}
                            
                            // Escape HTML characters
                            const char = tokens[i]
                                .replace(/&/g, '&amp;')
                                .replace(/</g, '&lt;')
                                .replace(/>/g, '&gt;');
                            
                            highlightedText += char;
                        }}
                        
                        if (currentHighlight) {{
                            highlightedText += '</span>';
                        }}
                        
                        responsePre.innerHTML = highlightedText;
                        div.appendChild(responsePre);
                        
                        resultsContainer.appendChild(div);
                    }});
                    
                    // Show results
                    document.getElementById('results').style.display = 'block';
                }} catch (error) {{
                    console.error('Error:', error);
                    alert('Error fetching activations. See console for details.');
                }} finally {{
                    document.getElementById('loading').style.display = 'none';
                }}
            }}
            
            // Function to load pre-computed activations
            function loadPrecomputedActivations(featureId) {{
                // This would be replaced with actual pre-computed data in a production version
                alert('Loading pre-computed activations for feature ' + featureId);
                // In a real implementation, this would load data from a JSON file or similar
            }}
        </script>
    </head>
    <body>
        <div class="container">
            <h1>Qwen Crosscoder Feature Explorer</h1>
            
            <div class="distribution-plot">
                <h2>Distribution of Decoder Norm Differences</h2>
                <img src="data:image/png;base64,{plot_data}" alt="Distribution Plot">
                <p>Negative values indicate base model features, positive values indicate R1 model features.</p>
            </div>
            
            <div class="search-box">
                <h2>Search for a Feature</h2>
                <p>Enter a feature ID to see its top activating examples:</p>
                <input type="number" id="featureId" min="0" max="{len(norm_diffs)-1}" placeholder="Feature ID">
                <button onclick="searchFeature()">Search</button>
                
                <div>
                    <h3>Quick Access:</h3>
                    <button onclick="document.getElementById('featureId').value={df_sorted.iloc[0].feature_id}; searchFeature()">Most Base-Exclusive</button>
                    <button onclick="document.getElementById('featureId').value={df_sorted.iloc[-1].feature_id}; searchFeature()">Most R1-Exclusive</button>
                    <button onclick="document.getElementById('featureId').value={df_sorted.iloc[len(df_sorted)//2].feature_id}; searchFeature()">Most Shared</button>
                </div>
            </div>
            
            <div id="loading" class="loading">
                <h2>Loading activations...</h2>
                <p>This may take a few moments.</p>
            </div>
            
            <div id="results" class="results">
                <div id="featureInfo" class="feature-info"></div>
                <h2>Top Activating Examples</h2>
                <div id="activationResults"></div>
            </div>
        </div>
        
        <script>
            // Server-side rendering of pre-computed activations
            const precomputedFeatures = [
                {df_sorted.iloc[0].feature_id},  // Most base-exclusive
                {df_sorted.iloc[-1].feature_id},  // Most R1-exclusive
                {df_sorted.iloc[len(df_sorted)//2].feature_id}  // Most shared
            ];
            
            // Pre-compute activations for these features
            precomputedFeatures.forEach(featureId => {{
                fetch('/precompute_activations?feature_id=' + featureId)
                    .then(response => response.json())
                    .then(data => console.log('Pre-computed activations for feature ' + featureId))
                    .catch(error => console.error('Error pre-computing activations:', error));
            }});
        </script>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open("feature_explorer.html", "w") as f:
        f.write(html_content)
    
    print("Generated HTML file: feature_explorer.html")
    
    # Create a simple server to handle activation requests
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import urlparse, parse_qs
    import threading
    
    # Cache for storing pre-computed activations
    activation_cache = {}
    
    class ActivationHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                parsed_url = urlparse(self.path)
                
                if parsed_url.path == '/get_activations':
                    query = parse_qs(parsed_url.query)
                    feature_id = int(query.get('feature_id', [0])[0])
                    
                    # Check if activations are in cache
                    if feature_id in activation_cache:
                        activations = activation_cache[feature_id]
                    else:
                        # Compute activations
                        activations = find_top_activating_tokens(feature_id)
                        activation_cache[feature_id] = activations
                    
                    # Send response
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(activations).encode())
                
                elif parsed_url.path == '/precompute_activations':
                    query = parse_qs(parsed_url.query)
                    feature_id = int(query.get('feature_id', [0])[0])
                    
                    # Start precomputation in a separate thread
                    threading.Thread(target=lambda: self.safe_precompute(feature_id)).start()
                    
                    # Send response
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "precomputing"}).encode())
                
                else:
                    # Serve the HTML file for any other path
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    with open("feature_explorer.html", "rb") as f:
                        self.wfile.write(f.read())
            except Exception as e:
                print(f"Error handling request: {e}")
                self.send_error(500, f"Internal server error: {str(e)}")
        
        def safe_precompute(self, feature_id):
            try:
                activation_cache[feature_id] = find_top_activating_tokens(feature_id)
            except Exception as e:
                print(f"Error precomputing activations for feature {feature_id}: {e}")
                activation_cache[feature_id] = {"error": str(e)}
    
    # Start the server
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, ActivationHandler)
    print("Starting server at http://localhost:8000")
    httpd.serve_forever()

if __name__ == "__main__":
    generate_html_file()