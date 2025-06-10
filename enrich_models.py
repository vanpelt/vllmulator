import requests
import json
from collections import defaultdict
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm
import re

FAMILY_MAP = {
    "deepseek-ai": "DeepSeek",
    "meta-llama": "Llama",
    "mistralai": "Mistral",
    "qwen": "Qwen",
    "microsoft": "Phi",
    "google": "Google",
    "ibm-granite": "IBM",
}
# Must-have models (add more details as needed)
# {"Family": "BGE", "Huggingface ID": "BAAI/bge-m3"},
MUST_HAVE_MODELS = [
    {"Family": "DeepSeek", "ID": "deepseek-ai/DeepSeek-R1-0528"},
    {"Family": "Llama", "ID": "meta-llama/Llama-3.3-70b-instruct"},
    {"Family": "Mistral", "ID": "mistralai/Mistral-Small-3.1-24B-Instruct-2503"},
    {"Family": "Llama", "ID": "meta-llama/Llama-4-Scout-17B-16E-Instruct"},
    {"Family": "Qwen", "ID": "Qwen/QwQ-32B"},
    {"Family": "Phi", "ID": "microsoft/Phi-4-mini-instruct"},
    {"Family": "Llama", "ID": "meta-llama/Llama-4-Maverick-17B-128E-Instruct"},
    {"Family": "Qwen", "ID": "Qwen/Qwen2.5-VL-7B-Instruct"},
    {"Family": "Granite", "ID": "ibm-granite/granite-3.3-8b-instruct"},
    {"Family": "Google", "ID": "google/gemma-3-27b-it"},
    {"Family": "Mistral", "ID": "mistralai/Mistral-Nemo-Instruct-2407"}
]

BERRIAI_JSON_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/litellm/model_prices_and_context_window_backup.json"

# Helper to get model metadata from HuggingFace
api = HfApi()
def detect_attention_optimizations(config, model_id):
    """Detect attention optimizations from model config"""
    optimizations = {
        "has_mla": False,
        "has_hybrid_attention": False, 
        "has_sliding_window": False,
        "has_gqa": False,
        "attention_details": {}
    }
    
    # Check for MLA (Multi-Head Latent Attention)
    # DeepSeek models with MLA have specific config keys
    if any(key in config for key in ["mla", "multi_head_latent_attention", "kv_lora_rank"]):
        optimizations["has_mla"] = True
        optimizations["attention_details"]["mla_compression_ratio"] = config.get("kv_lora_rank", "unknown")
    
    # Check model architecture names that use MLA
    if "deepseek" in model_id.lower() and any(v in model_id.lower() for v in ["v2", "v3", "r1"]):
        optimizations["has_mla"] = True
    
    # Check for Hybrid Local Attention patterns
    # Gemma 2 and some Llama models use sliding window + global attention
    sliding_window = config.get("sliding_window")
    attention_window_size = config.get("attention_window_size") 
    max_window_layers = config.get("max_window_layers")
    
    if sliding_window or attention_window_size or max_window_layers:
        optimizations["has_sliding_window"] = True
        optimizations["has_hybrid_attention"] = True  # Sliding window IS hybrid attention
        optimizations["attention_details"]["sliding_window_size"] = sliding_window or attention_window_size
        if max_window_layers:
            optimizations["attention_details"]["window_layers"] = max_window_layers
    
    # Gemma 2 and 3 use hybrid attention architectures
    if any(pattern in model_id.lower() for pattern in ["gemma-2", "gemma-3"]) or config.get("model_type") in ["gemma2", "gemma3"]:
        optimizations["has_hybrid_attention"] = True
        if "gemma-3" in model_id.lower() or config.get("model_type") == "gemma3":
            optimizations["attention_details"]["type"] = "gemma3_5to1_hybrid"
        else:
            optimizations["attention_details"]["type"] = "gemma2_hybrid"
    
    # Check for Grouped Query Attention (GQA)
    num_attention_heads = config.get("num_attention_heads", config.get("num_heads"))
    num_key_value_heads = config.get("num_key_value_heads") 
    
    if num_key_value_heads and num_attention_heads:
        if num_key_value_heads < num_attention_heads:
            optimizations["has_gqa"] = True
            optimizations["attention_details"]["gqa_ratio"] = num_attention_heads / num_key_value_heads
    
    # Some Llama models use RoPE scaling which can indicate optimizations
    rope_scaling = config.get("rope_scaling")
    if rope_scaling:
        scaling_type = rope_scaling.get("type", rope_scaling.get("rope_type"))
        if scaling_type in ["linear", "dynamic", "yarn", "longrope"]:
            optimizations["attention_details"]["rope_scaling"] = scaling_type
    
    # Check for attention layer patterns that suggest hybrid attention
    # Some models define different attention types per layer
    if any(key in config for key in ["attention_layers", "layer_types", "attention_config"]):
        optimizations["has_hybrid_attention"] = True
        optimizations["attention_details"]["type"] = "layer_specific"
    
    return optimizations

def get_hf_model_info(model_id, tokens = None):
    try:
        if tokens is None:
            tokens = {"max_input_tokens": 16385, "max_output_tokens": 4096}
        info = api.model_info(model_id)
        # Download config.json
        config_path = hf_hub_download(repo_id=model_id, filename="config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # Extract basic fields
        hidden_size = config.get("hidden_size")
        hidden_layers = config.get("num_hidden_layers")
        if not hidden_size:
            hidden_size = config.get("text_config", {}).get("hidden_size")
        if not hidden_layers:
            hidden_layers = config.get("text_config", {}).get("num_hidden_layers")
        parameters = info.safetensors.total
        if not parameters:
            parameters = parse_params_from_name(model_id)
        dtype = config.get("torch_dtype")
        model_type = config.get("model_type")
        prefix = model_id.split("/")[0]
        family = FAMILY_MAP.get(prefix, prefix)
        
        # Detect attention optimizations
        attention_opts = detect_attention_optimizations(config, model_id)
        
        result = {
            "Family": family,
            "ID": model_id,
            "Hidden Size": hidden_size,
            "Hidden Layers": hidden_layers,
            "Datatype": dtype,
            "Parameters": parameters,
            "Type": model_type,
            "Tags": info.tags,
            "Downloads": info.downloads,
            "Last Updated": info.last_modified.isoformat() if info.last_modified else None,
            "Likes": info.likes,
            "Input Tokens": tokens["max_input_tokens"],
            "Output Tokens": tokens["max_output_tokens"]
        }
        
        # Add attention optimization flags
        result.update(attention_opts)
        
        return result
    except Exception as e:
        print(f"Failed to get config for {model_id}: {e}")
        return None

def parse_params_from_name(model_id):
    # Look for patterns like 3B, 70B, 8M, etc.
    match = re.search(r'([0-9]+)([BM])', model_id, re.IGNORECASE)
    if match:
        num, scale = match.groups()
        num = int(num)
        if scale.upper() == 'B':
            return num * 1_000_000_000
        elif scale.upper() == 'M':
            return num * 1_000_000
    return None

def main():
    # Download the BerriAI model list
    resp = requests.get(BERRIAI_JSON_URL)
    resp.raise_for_status()
    model_dict = resp.json()

    # TODO: perhaps also grab most popular...
    # models = api.list_models(pipeline_tag="text-generation", limit=100)

    # Deduplicate by HuggingFace ID (case-insensitive)
    seen = set()
    unique_hf_ids = []
    tokens = {}
    for spec in model_dict.keys():
        # Remove provider prefix if present
        config = model_dict[spec]
        if config.get("mode") != "chat":
            continue
        parts = spec.split('/')
        if len(parts) > 1 and parts[0] in {"together", "nscale"}:
            hf_id = '/'.join(parts[1:])
        else:
            continue
        hf_id_lower = hf_id.lower()
        tokens[hf_id_lower] = {
            "max_input_tokens": config.get("max_input_tokens", 16385),
            "max_output_tokens": config.get("max_output_tokens", 4096),
        }
        if hf_id_lower not in seen:
            seen.add(hf_id_lower)
            unique_hf_ids.append(hf_id)

    # Add must-have models if missing
    for must in MUST_HAVE_MODELS:
        hf_id = must["ID"]
        if hf_id.lower() not in seen:
            unique_hf_ids.append(hf_id)
            seen.add(hf_id.lower())

    # Filter for models that exist on HuggingFace
    print(f"Checking {len(unique_hf_ids)} models on HuggingFace...")
    print(unique_hf_ids)

    # Limit to top 50 (by order in list)
    top_models = unique_hf_ids[:50]

    # Enrich with metadata
    enriched = []
    for model_id in tqdm(top_models, desc="Enriching models"):
        info = get_hf_model_info(model_id, tokens.get(model_id.lower()))
        print(info)
        if info:
            enriched.append(info)

    # Group by Family
    grouped = defaultdict(list)
    for model in enriched:
        grouped[model["Family"]].append(model)

    # Output JSON
    with open("enriched_models.json", "w") as f:
        json.dump(grouped, f, indent=2)
    print("Saved to enriched_models.json")

if __name__ == "__main__":
    main()