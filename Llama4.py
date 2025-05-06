import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import collections
import re
from typing import Tuple, Optional
import os

# --- Device Configuration ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)
print(f"Using device: {device}")

# For mixed precision training
try:
    import torch.cuda.amp as amp
    mixed_precision_available = torch.cuda.is_available()
    print(f"Mixed precision training available: {mixed_precision_available}")
except ImportError:
    mixed_precision_available = False
    print("Mixed precision training not available")

# --- BPE Tokenizer Implementation ---
class BPETokenizer:
    def __init__(self, corpus=None, num_merges=50):
        self.vocab = {}  # token to id
        self.inv_vocab = {}  # id to token
        self.merges = {}  # pair to merged token
        self.num_merges = num_merges
        self.end_of_word = "</w>"
        
        if corpus:
            self.train(corpus)
    
    def train(self, corpus):
        # Initialize with unique characters
        unique_chars = set()
        for text in corpus:
            for char in text:
                unique_chars.add(char)
        
        char_list = sorted(list(unique_chars))
        char_list.append(self.end_of_word)
        
        # Initialize vocabulary with characters
        for i, char in enumerate(char_list):
            self.vocab[char] = i
            self.inv_vocab[i] = char
        
        # Prepare initial word splits
        word_splits = {}
        for text in corpus:
            words = text.split()
            for word in words:
                if word:
                    char_list = list(word) + [self.end_of_word]
                    word_tuple = tuple(char_list)
                    if word_tuple not in word_splits:
                        word_splits[word_tuple] = 0
                    word_splits[word_tuple] += 1
        
        # BPE Algorithm
        current_splits = word_splits.copy()
        vocab_size = len(self.vocab)
        
        for i in range(self.num_merges):
            pair_counts = self._get_pair_stats(current_splits)
            if not pair_counts:
                break
                
            best_pair = max(pair_counts, key=pair_counts.get)
            new_token = best_pair[0] + best_pair[1]
            
            # Add to vocabulary
            self.vocab[new_token] = vocab_size
            self.inv_vocab[vocab_size] = new_token
            vocab_size += 1
            
            # Store merge rule
            self.merges[best_pair] = new_token
            
            # Apply merge
            current_splits = self._merge_pair(best_pair, current_splits)
    
    def _get_pair_stats(self, splits):
        pair_counts = collections.defaultdict(int)
        for word_tuple, freq in splits.items():
            symbols = list(word_tuple)
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i+1])
                pair_counts[pair] += freq
        return pair_counts
    
    def _merge_pair(self, pair_to_merge, splits):
        new_splits = {}
        first, second = pair_to_merge
        merged_token = first + second
        
        for word_tuple, freq in splits.items():
            symbols = list(word_tuple)
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == first and symbols[i+1] == second:
                    new_symbols.append(merged_token)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            new_splits[tuple(new_symbols)] = freq
        
        return new_splits
    
    def encode(self, text):
        words = text.split()
        tokens = []
        
        for word in words:
            symbols = list(word) + [self.end_of_word]
            
            # Apply merges based on learned rules
            while len(symbols) > 1:
                pairs = [(symbols[i], symbols[i+1]) for i in range(len(symbols)-1)]
                mergeable_pairs = [p for p in pairs if p in self.merges]
                
                if not mergeable_pairs:
                    break
                
                # Find first mergeable pair (greedy approach)
                first, second = mergeable_pairs[0]
                merged_token = self.merges[(first, second)]
                
                # Apply merge
                new_symbols = []
                i = 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and symbols[i] == first and symbols[i+1] == second:
                        new_symbols.append(merged_token)
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                symbols = new_symbols
            
            # Convert to token IDs
            for symbol in symbols:
                if symbol in self.vocab:
                    tokens.append(self.vocab[symbol])
                else:
                    # Handle unknown tokens
                    for char in symbol:
                        if char in self.vocab:
                            tokens.append(self.vocab[char])
        
        return tokens
    
    def decode(self, token_ids):
        text = ""
        for token_id in token_ids:
            token = self.inv_vocab.get(token_id, "[UNK]")
            if token == self.end_of_word:
                text += " "
            else:
                text += token
        return text.replace("</w>", " ").strip()

# --- Rotary Positional Embedding (RoPE) ---
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._set_cos_sin_cache(max_position_embeddings)
        
    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])
    
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
            
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...]
        )

# Apply Rotary Embeddings
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # q, k: [bs, num_heads, seq_len, head_dim]
    # cos, sin: [1, 1, seq_len, head_dim]
    # position_ids: [bs, seq_len]
    
    # Reshape position_ids to be compatible with the embeddings
    seq_len = position_ids.size(-1)
    
    # Get embeddings for the specific positions we need
    cos = cos[:, :, :seq_len, :]  # [1, 1, seq_len, head_dim]
    sin = sin[:, :, :seq_len, :]  # [1, 1, seq_len, head_dim]
    
    # Reshape q and k for applying rotary embeddings
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed

def rotate_half(x):
    # Rotate half the dimensions
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

# --- RMSNorm Layer ---
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)

# --- Expert MLP ---
class Expert(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout_rate=0.0, hidden_act="silu"):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = F.silu
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        gate = self.act_fn(gate)
        intermediate = gate * up
        intermediate = self.dropout(intermediate)
        return self.down_proj(intermediate)

# --- Mixture of Experts Layer ---
class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_experts = config["num_local_experts"]
        self.top_k = config["num_experts_per_tok"]
        self.intermediate_size = config["expert_dim"]
        self.dropout_rate = config.get("dropout_rate", 0.0)
        
        # Create experts
        self.experts = nn.ModuleList([
            Expert(self.hidden_size, self.intermediate_size, self.dropout_rate) 
            for _ in range(self.num_experts)
        ])
        
        # Router (gate) for selecting experts
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        
        # Load balancing coefficient
        self.router_z_loss_coef = 0.001
        
        # Shared Expert MLP
        self.shared_expert = Expert(
            self.hidden_size, 
            config["shared_expert_dim"], 
            self.dropout_rate
        )
        
    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_size = hidden_states.shape
        
        # Reshape for routing
        router_logits = self.router(hidden_states)  # [batch_size, sequence_length, num_experts]
        
        # Get routing probabilities
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Calculate auxiliary load balancing loss
        # This encourages balanced expert utilization during training
        if self.training:
            router_probs = routing_weights.mean(dim=[0, 1])  # Mean prob for each expert
            router_z_loss = torch.sum(router_probs * torch.log(router_probs * self.num_experts + 1e-6))
        else:
            router_z_loss = 0.0
        
        # Get top-k experts for each token
        routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        # Expand for gather operations
        hidden_states = hidden_states.view(-1, hidden_size)  # [batch_size*sequence_length, hidden_size]
        
        # Initialize output
        expert_outputs = torch.zeros_like(hidden_states)
        
        # Process each token through its selected experts with improved batching
        flat_selected_experts = selected_experts.view(-1, self.top_k)  # [batch_size*sequence_length, top_k]
        flat_routing_weights = routing_weights.view(-1, self.top_k)    # [batch_size*sequence_length, top_k]
        
        # Process tokens with each expert
        for expert_idx in range(self.num_experts):
            # Find which positions use this expert
            expert_mask = (flat_selected_experts == expert_idx)
            if not expert_mask.any():
                continue
                
            # Get indices of tokens going to this expert
            token_indices = torch.nonzero(expert_mask, as_tuple=True)[0]
            
            # Get the corresponding routing weights
            weight_indices = torch.nonzero(expert_mask, as_tuple=True)
            expert_weights = flat_routing_weights[weight_indices]
            
            # Get inputs for this expert
            expert_inputs = hidden_states[token_indices]
            
            # Process through the expert
            expert_output = self.experts[expert_idx](expert_inputs)
            
            # Scale by the routing weights
            expert_output = expert_output * expert_weights.unsqueeze(-1)
            
            # Add to the output at the right positions
            expert_outputs.index_add_(0, token_indices, expert_output)
        
        # Add shared expert output
        shared_output = self.shared_expert(hidden_states.view(batch_size, sequence_length, hidden_size))
        
        # Final output combining experts and shared
        final_output = expert_outputs.view(batch_size, sequence_length, hidden_size) + shared_output
        
        # Include auxiliary load balancing loss
        return final_output, router_z_loss * self.router_z_loss_coef

# --- Multi-head Attention with GQA ---
class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_heads"]
        self.head_dim = config["head_dim"]
        self.num_key_value_heads = config["num_key_value_heads"]
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.dropout_rate = config.get("dropout_rate", 0.0)
        
        # Projection matrices
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # Attention dropout
        self.attn_dropout = nn.Dropout(self.dropout_rate)
        self.resid_dropout = nn.Dropout(self.dropout_rate)
        
        # Rotary position embeddings
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=config["max_position_embeddings"])
        
        # Optional L2 norm for Q and K
        self.use_qk_norm = config.get("use_qk_norm", False)
        if self.use_qk_norm:
            self.q_norm = L2Norm()
            self.k_norm = L2Norm()

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        batch_size, seq_length, _ = hidden_states.shape
        
        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape: [batch_size, seq_length, num_heads, head_dim]
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Get rotary embeddings
        cos, sin = self.rotary_emb(value_states, seq_length)
        
        # Apply rotary embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
        
        # Apply L2 norm if configured
        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)
        
        # Handle past key values (for inference)
        if past_key_value is not None:
            # Concatenate past keys and values
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
            
        # Save current key/value states if needed for auto-regressive generation
        past_key_value = (key_states, value_states)
        
        # Repeat K and V for Grouped-Query Attention
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)
        
        # Calculate attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax over sequence dimension
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Apply attention dropout during training
        attn_weights = self.attn_dropout(attn_weights)
        
        # Get context from attention
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and project back to hidden size
        attn_output = attn_output.transpose(1, 2).contiguous()
        # Use the actual sequence length from attention output instead of input seq_length
        output_seq_length = attn_output.size(1)
        attn_output = attn_output.reshape(batch_size, output_seq_length, self.num_heads * self.head_dim)
        
        attn_output = self.o_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        return attn_output, past_key_value
    
    def _repeat_kv(self, hidden_states, n_rep):
        """Repeat key/value states for Grouped-Query Attention"""
        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, n_rep, seq_len, head_dim
        )
        return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)

# --- Optional L2 Norm ---
class L2Norm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

# --- Llama 4 Decoder Layer ---
class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.dropout_rate = config.get("dropout_rate", 0.0)
        
        # Pre-normalization layers
        self.input_layernorm = RMSNorm(self.hidden_size, eps=config["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=config["rms_norm_eps"])
        
        # Self-attention
        self.self_attn = Attention(config)
        
        # MoE FFN or standard FFN based on layer configuration
        use_moe = config.get("use_moe", False)
        moe_layers = config.get("moe_layers", [])
        if use_moe and (moe_layers is None or config["layer_idx"] in moe_layers):
            self.mlp = MoE(config)
            self.has_moe = True
        else:
            # Standard FFN if not using MoE for this layer
            self.mlp = Expert(self.hidden_size, config["intermediate_size"], self.dropout_rate)
            self.has_moe = False
        
    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        # Self-attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        attn_output, past_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value
        )
        
        # First residual connection
        hidden_states = residual + attn_output
        
        # FFN/MoE block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        if self.has_moe:
            hidden_states, router_z_loss = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
            router_z_loss = 0.0
        
        # Second residual connection
        hidden_states = residual + hidden_states
        
        if self.has_moe:
            return hidden_states, past_key_value, router_z_loss
        else:
            return hidden_states, past_key_value, 0.0

# --- Llama 4 Model ---
class Llama4Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_size"]
        self.max_position_embeddings = config["max_position_embeddings"]
        self.dropout_rate = config.get("dropout_rate", 0.0)
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.embed_dropout = nn.Dropout(self.dropout_rate)
        
        # Layers
        self.layers = nn.ModuleList()
        for i in range(config["num_layers"]):
            layer_config = dict(config)
            layer_config["layer_idx"] = i
            self.layers.append(DecoderLayer(layer_config))
        
        # Final normalization layer
        self.norm = RMSNorm(self.hidden_size, eps=config["rms_norm_eps"])
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        std = self.config.get("initializer_range", 0.02)
        
        if isinstance(module, nn.Linear):
            # Use normal distribution initialization for projection matrices
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            
    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None):
        batch_size, seq_length = input_ids.shape
        
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.embed_dropout(hidden_states)
        
        # Generate positional IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
        
        # Initialize or use provided past key values
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        
        # Create causal mask if attention mask not provided
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), device=hidden_states.device
            )
        
        # Convert attention mask to 4D causal mask
        causal_mask = self._prepare_decoder_attention_mask(
            attention_mask, 
            (batch_size, seq_length),
            hidden_states.device,
            past_key_values
        )
        
        # Process through decoder layers
        present_key_values = []
        
        # Collect router z losses from MoE layers
        all_router_z_losses = []
        
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i]
            hidden_states, present_kv, router_z_loss = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_kv
            )
            present_key_values.append(present_kv)
            all_router_z_losses.append(router_z_loss)
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        # Sum all router losses
        router_z_loss = sum(all_router_z_losses)
        
        return hidden_states, present_key_values, router_z_loss
    
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, device, past_key_values=None):
        # Calculate total sequence length including past
        if past_key_values is not None and past_key_values[0] is not None:
            past_length = past_key_values[0][0].shape[2]  # past_key's seq length
        else:
            past_length = 0
            
        seq_length = input_shape[1] + past_length
        
        # Create causal mask
        # [batch_size, 1, target_length, source_length]
        causal_mask = torch.zeros(
            (input_shape[0], 1, input_shape[1], seq_length), 
            device=device,
            dtype=torch.float32
        )
        
        # Lower triangular mask (including the diagonal)
        causal_mask = torch.tril(
            torch.ones((input_shape[0], 1, input_shape[1], seq_length), device=device)
        )
        
        # If attention_mask is provided, apply it
        if attention_mask is not None:
            # Adjust attention mask to match causal mask dimensions
            expanded_attn_mask = attention_mask[:, None, None, :].expand(
                input_shape[0], 1, input_shape[1], attention_mask.size(1)
            )
            # Ensure sizes match before multiplying
            if expanded_attn_mask.size(-1) == causal_mask.size(-1):
                causal_mask = causal_mask * expanded_attn_mask
        
        # Convert to additive mask (0 for attended positions, -inf for masked)
        causal_mask = causal_mask.masked_fill(causal_mask == 0, -1e9)
        
        return causal_mask

# --- Llama 4 For Causal LM ---
class Llama4ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Llama4Model(config)
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Optional weight tying between embedding and LM head
        if config.get("tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight
            
    def _init_weights(self, module):
        std = self.config.get("initializer_range", 0.02)
        
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            
    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values
        )
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        router_z_loss = outputs[2]
        
        # Return both logits and aux losses for training
        return logits, outputs[1], router_z_loss  # logits, present_key_values, router_z_loss
    
    def generate(self, input_ids, max_length=50, temperature=1.0, top_k=50, top_p=0.9, repetition_penalty=1.2, do_sample=True):
        """
        Enhanced text generation with better sampling strategies.
        
        Args:
            input_ids: Starting token ids
            max_length: Maximum length to generate
            temperature: Temperature for sampling (higher = more random)
            top_k: Number of highest probability tokens to keep (0 = disabled)
            top_p: Nucleus sampling probability threshold (1.0 = disabled)
            repetition_penalty: Penalty for repeating tokens (1.0 = no penalty)
            do_sample: Whether to sample or use greedy decoding
        """
        batch_size = input_ids.shape[0]
        past_key_values = None
        past_length = 0
        
        # Set model to eval mode for generation
        self.eval()
        
        # Store initial input length to ensure we generate new tokens
        initial_length = input_ids.shape[1]
        
        # Generate until we reach max_length or the context limit
        while input_ids.shape[1] < max_length:
            # Forward pass
            seq_length = input_ids.shape[1]
            
            # Create position ids - handle past key values for proper positioning
            if past_key_values is not None and past_key_values[0] is not None:
                past_length = past_key_values[0][0].shape[2]
                # Position ids for the current token need to account for past context
                position_ids = torch.arange(past_length, past_length + seq_length, 
                                          device=input_ids.device).unsqueeze(0)
            else:
                position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
            
            # Get just the last token for efficiency in generation
            if past_key_values is not None:
                # Only need to process the last token with KV cache
                current_input_ids = input_ids[:, -1:]
                current_position_ids = position_ids[:, -1:]
            else:
                # First call - need to process the entire input
                current_input_ids = input_ids
                current_position_ids = position_ids
            
            with torch.no_grad():
                # Forward pass with the model
                logits, past_key_values, _ = self.forward(
                    current_input_ids,
                    position_ids=current_position_ids,
                    past_key_values=past_key_values,
                    attention_mask=None  # Skip attention mask for generation
                )
            
            # Get logits for the next token
            next_token_logits = logits[:, -1, :].clone()
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty - reduce probability of tokens that have already appeared
            if repetition_penalty > 1.0:
                for b in range(batch_size):
                    for token_id in set(input_ids[b].tolist()):
                        if next_token_logits[b, token_id] < 0:
                            # If the token logit is negative, make it more negative 
                            next_token_logits[b, token_id] *= repetition_penalty
                        else:
                            # If the token logit is positive, make it less positive
                            next_token_logits[b, token_id] /= repetition_penalty
            
            if do_sample:
                # Apply top-k sampling
                if top_k > 0:
                    # Keep only the top-k tokens
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    # Create a mask for tokens to keep
                    next_token_logits = next_token_logits.masked_fill(
                        ~torch.zeros_like(next_token_logits, dtype=torch.bool).scatter_(1, top_k_indices, 1), 
                        -float('Inf')
                    )
                
                # Apply nucleus (top-p) sampling
                if top_p < 1.0:
                    # Sort logits in descending order
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    # Calculate cumulative probabilities
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Create a mask for tokens with cumulative probability > top_p
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the mask to keep at least one token
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0  # Keep the first token
                    
                    # Convert sorted indices mask back to original logits space
                    for batch_idx in range(batch_size):
                        indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                        next_token_logits[batch_idx, indices_to_remove] = -float('Inf')
                
                # Sample from the filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Check for NaNs in probs
                if torch.isnan(probs).any():
                    # Handle case where distribution has NaNs by falling back to greedy
                    print("Warning: NaN detected in sampling distribution, falling back to greedy decoding")
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                else:
                    try:
                        next_token = torch.multinomial(probs, num_samples=1)
                    except:
                        # If sampling fails, fall back to greedy
                        print("Warning: Sampling failed, falling back to greedy decoding")
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append the token to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Debug print if needed
            # print(f"Generated token: {next_token.item()}, Current length: {input_ids.shape[1]}")
            
            # Stop if we've generated enough text (or if we hit a stopping condition like EOS)
            if input_ids.shape[1] >= max_length:
                break
        
        return input_ids

# --- Training corpus and main script ---
def main():
    print("\n*** ENHANCED LLAMA 4 MODEL IMPLEMENTATION ***\n")
    
    # Expanded CRISPR training corpus
    corpus = [
        "CRISPR is a gene-editing technology derived from a natural defense system in bacteria. It allows scientists to cut DNA at specific locations using a protein called Cas9 guided by RNA. By designing a guide RNA that matches the target gene, researchers can insert, delete, or modify genetic material with high precision. This method has revolutionized molecular biology, enabling faster and more accurate genetic studies. However, ethical concerns remain, especially regarding potential human germline editing.",
        "The CRISPR-Cas9 system functions by introducing double-strand breaks at specific genomic sites. These breaks are then repaired by the cell's natural DNA repair mechanisms. Researchers exploit this process to disrupt gene function or introduce new genetic sequences. Applications of CRISPR include creating disease models, correcting genetic mutations, and engineering crops with improved traits.",
        "CRISPR technology consists of two key components: the Cas9 enzyme and the guide RNA. The guide RNA contains a sequence that matches the target DNA, directing the Cas9 protein to the precise location. Once there, Cas9 acts like molecular scissors to cut both strands of the DNA helix.",
        "Scientists can modify the CRISPR system to perform different functions beyond cutting DNA. By using modified versions of Cas9, researchers can regulate gene expression, visualize specific DNA sequences, or make precise single-base edits without cutting the DNA strands completely.",
        "CRISPR has significant advantages over previous gene-editing technologies. It's more precise, efficient, cost-effective, and easier to use than methods like zinc-finger nucleases or TALENs. These properties have accelerated the pace of genetic research across the biological sciences.",
        "In medicine, CRISPR shows promise for treating genetic disorders like sickle cell disease, cystic fibrosis, and Huntington's disease. Clinical trials are underway to test CRISPR-based therapies for various conditions, with early results showing potential for safe and effective genetic intervention.",
        "Agricultural applications of CRISPR include developing crops with enhanced nutritional profiles, improved disease resistance, and better adaptation to climate change. Unlike traditional GMO approaches, CRISPR can sometimes create changes indistinguishable from natural mutations.",
        "The ethics of CRISPR technology raises important questions about human genetic modification. The scientific community has called for careful regulation, especially regarding germline editing that would affect future generations. International scientific bodies are working to establish guidelines for responsible research.",
        "CRISPR-Cas9 was adapted from a naturally occurring defense mechanism in bacteria. Bacteria use CRISPR sequences to remember viral DNA and defend against repeated attacks. When a virus infects a bacterium, the CRISPR system captures fragments of the viral DNA and inserts them into the bacterial genome as CRISPR arrays.",
        "The discovery of CRISPR's gene-editing potential has led to a scientific revolution. Several researchers played key roles in this development, including Jennifer Doudna and Emmanuelle Charpentier, who won the 2020 Nobel Prize in Chemistry for their pioneering work. Their research demonstrated how CRISPR-Cas9 could be programmed to cut DNA at specific sites.",
        "Beyond Cas9, researchers have discovered other CRISPR-associated proteins like Cas12 and Cas13 with unique capabilities. Cas13, for example, targets RNA instead of DNA, opening up possibilities for treating RNA-based diseases and developing diagnostic tools. The CRISPR toolkit continues to expand as scientists identify new variants with specialized functions.",
        "CRISPR technology has accelerated the pace of biological research dramatically. Experiments that once took months or years can now be completed in weeks. This efficiency has democratized gene editing, allowing smaller labs and institutions to conduct advanced genetic research without massive funding or specialized equipment.",
        # New content for expanded corpus
        "CRISPR-Cas9 gene editing technology continues to evolve rapidly as researchers develop new applications and refinements. Recent innovations include base editing, which allows scientists to change individual DNA bases without making double-strand breaks, and prime editing, which offers even more precise control over genetic modifications.",
        "The therapeutic potential of CRISPR is being explored in clinical trials for conditions like sickle cell disease, beta-thalassemia, and certain forms of cancer. Early results from these trials suggest that CRISPR-based treatments can be both safe and effective, though long-term outcomes remain to be seen.",
        "Beyond medicine, CRISPR is transforming agriculture and food production. Scientists are using the technology to develop crops with enhanced nutritional profiles, improved disease resistance, and better tolerance to environmental stresses like drought and heat. Unlike traditional GMOs, CRISPR-edited plants often contain genetic changes indistinguishable from those that might occur naturally.",
        "In the environmental field, researchers are exploring 'gene drives' that could potentially eliminate disease vectors like malaria-carrying mosquitoes or control invasive species. These applications raise complex ecological and ethical questions about humanity's role in reshaping natural systems.",
        "The accessibility of CRISPR technology has democratized genetic research, allowing smaller labs and institutions to conduct sophisticated experiments that were previously possible only at well-funded research centers. This democratization has accelerated the pace of discovery but also raised concerns about potential misuse.",
        "Regulatory frameworks for CRISPR applications vary widely around the world. Some countries have embraced the technology with minimal restrictions, while others have implemented more cautious approaches, particularly regarding human germline editing that would affect future generations.",
        "As CRISPR technology matures, researchers continue to discover new Cas proteins with unique properties and capabilities. These expanded toolkits offer solutions to previous limitations, enabling more precise editing across a broader range of cell types and genomic contexts.",
        "The ethical discourse surrounding CRISPR has evolved beyond initial concerns about designer babies to encompass questions of equity, access, and cultural values. Scientists, ethicists, policymakers, and the public are engaged in ongoing conversations about how to harness the potential benefits of gene editing while minimizing risks and respecting diverse perspectives.",
        "Education about CRISPR science and its implications has become increasingly important as the technology touches more aspects of society. Schools, universities, and public science initiatives are developing curricula and outreach programs to foster scientific literacy and informed citizen participation in decisions about the future of gene editing.",
        "Looking ahead, CRISPR technology is likely to become more integrated into routine healthcare, agriculture, and environmental management. The challenge for society will be to navigate this transition thoughtfully, establishing norms and practices that maximize benefits while addressing concerns about safety, equity, and unintended consequences.",
        "The molecular mechanism of CRISPR involves the Cas9 protein forming a complex with guide RNA, which then binds to complementary DNA sequences. Upon binding, the Cas9 protein undergoes a conformational change that activates its nuclease domains, allowing it to cleave both strands of the target DNA.",
        "CRISPR systems in nature are classified into different types (I through VI) based on their components and mechanisms. The type II system, which includes Cas9, is the most widely used for gene editing applications due to its simplicity and efficiency.",
        "Scientists are developing methods to improve the specificity of CRISPR-Cas9 and reduce off-target effects, which occur when the system edits unintended DNA sequences. Strategies include engineering more precise Cas9 variants, optimizing guide RNA design, and using paired nickases that make single-strand cuts instead of double-strand breaks.",
        "Beyond gene editing, CRISPR technology is being used to develop diagnostic tools for detecting pathogens and diseases. Systems based on Cas12 and Cas13 can be programmed to recognize specific nucleic acid sequences from viruses or bacteria, offering rapid and sensitive diagnostic capabilities.",
        "The field of epigenome editing uses modified CRISPR systems with deactivated nuclease domains (dCas9) fused to effector proteins that can alter gene expression without changing the underlying DNA sequence. This approach allows researchers to turn genes on or off temporarily, offering new ways to study gene function and potentially treat diseases.",
        "CRISPR technology is advancing our understanding of genetic diseases by enabling the creation of more accurate animal models that reflect human conditions. These models help researchers identify disease mechanisms and test potential therapies before moving to clinical trials.",
        "The combination of CRISPR with other technologies like single-cell sequencing and artificial intelligence is accelerating the pace of genomic research. These integrated approaches allow scientists to analyze the effects of genetic modifications at unprecedented resolution and scale.",
        "Ethical frameworks for CRISPR applications emphasize principles such as beneficence, non-maleficence, justice, and respect for autonomy. These principles guide decisions about when and how gene editing should be used, particularly for applications that could affect future generations.",
        "The potential use of CRISPR for enhancing human traits beyond treating disease raises profound questions about human identity, diversity, and the boundary between therapy and enhancement. These questions involve not only scientific and medical considerations but also philosophical and cultural perspectives.",
        "International collaboration is essential for establishing responsible governance of CRISPR technology. Initiatives like the International Summit on Human Genome Editing bring together experts from various disciplines and countries to develop guidelines and standards for the field."
    ]
    
    # Try to load additional text data if available
    try:
        additional_corpus_file = "additional_corpus.txt"
        if os.path.exists(additional_corpus_file):
            print(f"Loading additional corpus data from {additional_corpus_file}")
            with open(additional_corpus_file, 'r') as f:
                additional_texts = f.read().split('\n\n')  # Split by paragraphs
                # Filter out empty or very short paragraphs
                additional_texts = [text.strip() for text in additional_texts if len(text.strip()) > 100]
                corpus.extend(additional_texts)
                print(f"Added {len(additional_texts)} paragraphs from additional corpus")
    except Exception as e:
        print(f"Could not load additional corpus: {e}")
    
    print(f"Total corpus size: {len(corpus)} paragraphs")
    
    # Create and train tokenizer with more merges
    print("Training tokenizer...")
    tokenizer = BPETokenizer(corpus=corpus, num_merges=1000)  # Increased from 500
    vocab_size = len(tokenizer.vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # Tokenize the corpus
    tokenized_corpus = [torch.tensor(tokenizer.encode(text)) for text in corpus]
    
    # Create dataset
    max_length = 512  # Increased from 256
    
    # Generate train data from tokens with more sophisticated augmentation
    def get_batch(batch_size=8, context_length=128):  # Increased context length
        # Randomly select texts from corpus with replacement
        text_idx = torch.randint(len(tokenized_corpus), (batch_size,))
        x = []
        y = []
        
        for i in range(batch_size):
            text = tokenized_corpus[text_idx[i]]
            if len(text) <= context_length:
                # Pad with zeros if text is shorter than context_length
                padded = torch.zeros(context_length, dtype=torch.long)
                padded[:len(text)] = text
                start_idx = 0
            else:
                # Random starting position if text is longer
                start_idx = torch.randint(0, len(text) - context_length, (1,)).item()
                padded = text[start_idx:start_idx + context_length]
            
            x.append(padded)
            # For targets, shift input by one position (predict next token)
            target = torch.zeros(context_length, dtype=torch.long)
            if start_idx + context_length < len(text):
                target[:context_length] = text[start_idx+1:start_idx+context_length+1]
            else:
                target[:len(text)-start_idx-1] = text[start_idx+1:]
            y.append(target)
        
        x = torch.stack(x)
        y = torch.stack(y)
        x, y = x.to(device), y.to(device)
        return x, y
    
    # Enhanced model configuration - adjust size based on device capabilities
    if device == 'cpu':
        # Smaller configuration for CPU training
        config = {
            "vocab_size": vocab_size,
            "hidden_size": 384,        # Reduced from 768
            "num_layers": 6,           # Reduced from 12
            "num_heads": 12,           # Reduced from 24
            "head_dim": 32,
            "num_key_value_heads": 4,  # Reduced from 8
            "max_position_embeddings": max_length,
            "rms_norm_eps": 1e-5,
            "initializer_range": 0.02,
            "use_qk_norm": True,
            "use_moe": True,
            "moe_layers": [1, 3, 5],   # Updated for 6 layers
            "num_local_experts": 4,    # Reduced from 8
            "num_experts_per_tok": 2,
            "expert_dim": 1024,        # Reduced from 2048
            "shared_expert_dim": 1024, # Reduced from 2048
            "intermediate_size": 1024, # Reduced from 2048
            "dropout_rate": 0.1,
        }
        print("Using smaller model configuration for CPU training")
    else:
        # 1B parameter configuration for GPU
        config = {
            "vocab_size": vocab_size,
            "hidden_size": 2048,       # Increased for 1B model
            "num_layers": 24,          # Increased for 1B model
            "num_heads": 32,           # Increased for 1B model
            "head_dim": 64,            # Increased head dimension
            "num_key_value_heads": 8,  # GQA with 4x reduction
            "max_position_embeddings": max_length,
            "rms_norm_eps": 1e-5,
            "initializer_range": 0.02,
            "use_qk_norm": True,
            "use_moe": True,
            "moe_layers": [i for i in range(24) if i % 3 == 0],  # Every third layer is MoE
            "num_local_experts": 16,   # More experts for 1B model
            "num_experts_per_tok": 2,
            "expert_dim": 5120,        # Increased expert dimension
            "shared_expert_dim": 5120, # Increased shared expert dimension
            "intermediate_size": 5120, # Increased intermediate size
            "dropout_rate": 0.1,
        }
        print("Using 1B parameter configuration for GPU training")
        
    # Calculate and print approximate parameter count before initialization
    if device != 'cpu':
        # Roughly estimate parameter count for large model
        non_moe_layers = config["num_layers"] - len(config["moe_layers"])
        moe_layers = len(config["moe_layers"])
        
        # Attention params
        attn_params = config["num_layers"] * (
            # QKV projections
            3 * config["hidden_size"] * config["hidden_size"] + 
            # Output projection
            config["hidden_size"] * config["hidden_size"]
        )
        
        # MoE params
        moe_params = moe_layers * (
            # Experts
            config["num_local_experts"] * (
                # Up/gate projections
                2 * config["hidden_size"] * config["expert_dim"] +
                # Down projection
                config["expert_dim"] * config["hidden_size"]
            ) +
            # Shared expert
            (2 * config["hidden_size"] * config["shared_expert_dim"] +
             config["shared_expert_dim"] * config["hidden_size"]) +
            # Router
            config["hidden_size"] * config["num_local_experts"]
        )
        
        # FFN params for non-MoE layers
        ffn_params = non_moe_layers * (
            # Up/gate projections
            2 * config["hidden_size"] * config["intermediate_size"] +
            # Down projection
            config["intermediate_size"] * config["hidden_size"]
        )
        
        # Other params (embeddings, layer norms, etc.)
        other_params = (
            # Token embeddings
            config["vocab_size"] * config["hidden_size"] +
            # Layer norms (2 per layer)
            2 * config["num_layers"] * config["hidden_size"] +
            # Final layer norm
            config["hidden_size"] +
            # LM head
            config["hidden_size"] * config["vocab_size"]
        )
        
        total_params = attn_params + moe_params + ffn_params + other_params
        print(f"Estimated parameter count: {total_params:,} parameters")
        
        # Check if this is reasonable for training
        if device == 'cuda' and torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            print(f"Available GPU memory: {gpu_mem / 1e9:.2f} GB")
            
            # Rough estimate: 4 bytes per parameter, 1.5x for optimizer states
            mem_needed = total_params * 4 * 1.5 / 1e9
            print(f"Estimated memory required: {mem_needed:.2f} GB")
            
            if mem_needed > gpu_mem / 1e9:
                print(f"Warning: Model may not fit in GPU memory. Consider reducing size.")
    
    print("Initializing model...")
    model = Llama4ForCausalLM(config).to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
    
    # Improved training parameters
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=2e-4,            # Lower initial learning rate
        weight_decay=0.1,   # Increased weight decay
        betas=(0.9, 0.95)   # Beta parameters
    )
    
    # Better learning rate scheduler with warmup
    def get_lr_scheduler(optimizer, warmup_steps=100, max_steps=1000):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(max_steps - current_step) / float(max(1, max_steps - warmup_steps)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training hyperparameters
    if device == 'cpu':
        num_epochs = 100         # Fewer epochs for CPU
        patience = 5             # Early stopping patience
    else:
        num_epochs = 200         # Adjusted for larger model
        patience = 10            # More patience for 1B model
        
    corpus_size = len(tokenized_corpus)
    batch_size = min(4, corpus_size) if device == 'cpu' else min(2, corpus_size)  # Smaller batch for GPU due to memory
    context_length = 128     # Increased context length
    gradient_accumulation_steps = 16 if device != 'cpu' else 4  # More accumulation steps for large model
    max_grad_norm = 1.0      # For gradient clipping
    
    # Create scheduler
    num_batches_per_epoch = max(1, corpus_size // batch_size)  # Ensure at least 1 batch
    total_steps = num_batches_per_epoch * num_epochs
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)
    
    # Set up mixed precision training if available
    mixed_precision_enabled = mixed_precision_available and device == 'cuda'
    scaler = amp.GradScaler() if mixed_precision_enabled else None
    
    if device != 'cpu':
        print("Enabling mixed precision training for faster computation")
    
    # Training loop with improvements
    print("Starting training...")
    print(f"Corpus size: {corpus_size}, Batch size: {batch_size}, Batches per epoch: {num_batches_per_epoch}")
    print(f"Training for up to {num_epochs} epochs with early stopping patience of {patience}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps} (effective batch size: {batch_size * gradient_accumulation_steps})")
    step = 0
    best_loss = float('inf')
    no_improvement_count = 0
    
    # Allow checkpointing to resume training
    start_epoch = 0
    checkpoint_path = 'llama4_checkpoint_latest.pt'
    
    # Check if checkpoint exists to resume
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint at {checkpoint_path}, resuming training...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        step = checkpoint['step']
        print(f"Resuming from epoch {start_epoch} with best loss {best_loss:.4f}")
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        
        # Training with gradient accumulation
        for i in range(0, corpus_size, batch_size):
            step += 1
            
            try:
                # Get batch
                xb, yb = get_batch(batch_size, context_length)
                
                # Mixed precision training
                if mixed_precision_enabled:
                    with amp.autocast():
                        # Forward pass with router z-loss
                        logits, _, router_z_loss = model(xb)
                        # Main loss - cross entropy
                        ce_loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
                        # Total loss including router balancing
                        loss = ce_loss + router_z_loss
                        # Scale loss for gradient accumulation
                        loss = loss / gradient_accumulation_steps
                    
                    # Backward pass with scaled gradients
                    scaler.scale(loss).backward()
                    
                    if step % gradient_accumulation_steps == 0:
                        # Unscale gradients for clipping
                        scaler.unscale_(optimizer)
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        # Update weights with the scaler
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        scheduler.step()
                else:
                    # Standard training without mixed precision
                    # Forward pass with router z-loss
                    logits, _, router_z_loss = model(xb)
                    # Main loss - cross entropy
                    ce_loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
                    # Total loss including router balancing
                    loss = ce_loss + router_z_loss
                    # Scale loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps
                    
                    # Backward pass
                    loss.backward()
                    
                    if step % gradient_accumulation_steps == 0:
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        # Update weights
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                
                epoch_loss += loss.item() * gradient_accumulation_steps
                
                # Log progress
                if step % 10 == 0:
                    lr = scheduler.get_last_lr()[0]
                    print(f"Epoch {epoch}, Step {step}, CE Loss: {ce_loss.item():.4f}, Z Loss: {router_z_loss.item():.4f}, Total: {loss.item() * gradient_accumulation_steps:.4f}, LR: {lr:.6f}")
                
                # Memory management for GPU
                if device == 'cuda':
                    # Clear cache periodically to avoid memory fragmentation
                    if step % 50 == 0:
                        torch.cuda.empty_cache()
                    
                    # Log GPU memory usage
                    if step % 100 == 0:
                        allocated = torch.cuda.memory_allocated() / 1e9
                        max_allocated = torch.cuda.max_memory_allocated() / 1e9
                        print(f"GPU Memory: {allocated:.2f} GB (Max: {max_allocated:.2f} GB)")
            
            except Exception as e:
                print(f"Error during training step: {e}")
                # Try to recover and continue
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue
        
        # Average loss for the epoch - safely calculate the average
        avg_epoch_loss = epoch_loss / num_batches_per_epoch
        print(f"Epoch {epoch} complete. Average Loss: {avg_epoch_loss:.4f}")
        
        # Save latest checkpoint every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_epoch_loss,
            'config': config,
            'step': step
        }, 'llama4_checkpoint_latest.pt')
        
        # Save checkpoint if it's the best model so far
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            print(f"New best model with loss: {best_loss:.4f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
                'config': config,
                'step': step
            }, 'llama4_best_model.pt')
            no_improvement_count = 0  # Reset counter
        else:
            no_improvement_count += 1
            print(f"No improvement for {no_improvement_count} epochs")
            
            # Early stopping
            if no_improvement_count >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save periodic checkpoint every 5 epochs
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_epoch_loss,
                'config': config,
                'step': step
            }, f'llama4_checkpoint_epoch_{epoch}.pt')
    
    print("\nTraining complete!")
    
    # Save the final model
    print("Saving final model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, 'llama4_enhanced_model.pt')
    
    # Save tokenizer separately
    import pickle
    with open('llama4_enhanced_tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    print("Final model saved to llama4_enhanced_model.pt")
    print("Tokenizer saved to llama4_enhanced_tokenizer.pkl")
    
    # Generate text with improved sampling
    print("\nGenerating text examples...")
    model.eval()  # Set to evaluation mode
    
    prompts = [
        "CRISPR technology",
        "Gene editing",
        "The Cas9 protein",
        "DNA modification"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
        
        # Multiple generation attempts with different parameters
        generation_params = [
            {"temp": 0.7, "top_k": 50, "top_p": 0.9, "repetition_penalty": 1.2},
            {"temp": 0.8, "top_k": 40, "top_p": 0.95, "repetition_penalty": 1.3},
            {"temp": 0.9, "top_k": 0, "top_p": 0.85, "repetition_penalty": 1.1}
        ]
        
        for params in generation_params:
            generated_ids = model.generate(
                input_ids, 
                max_length=150,  # Longer generations 
                temperature=params["temp"], 
                top_k=params["top_k"],
                top_p=params["top_p"],
                repetition_penalty=params["repetition_penalty"]
            )
            generated_text = tokenizer.decode(generated_ids[0].tolist())
            print(f"  Settings (T={params['temp']}, K={params['top_k']}, P={params['top_p']}): {generated_text}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()