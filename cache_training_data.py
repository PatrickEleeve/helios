# helios/cache_training_data.py (新文件)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
import os
from typing import List, Dict, Any

from configs.config import BASE_MODEL_NAME, TARGET_COMM_LAYER, DEVICE

# --- 加载与 train_adapters.py 完全相同的模型和分词器 ---
print("Loading base model and tokenizer for caching...")
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=dtype
).to(DEVICE)
base_model.eval()

# --- 几乎原封不动地复制辅助函数 ---
@torch.no_grad()
def get_semantic_vector(text: str) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
    outputs = base_model(**inputs, output_hidden_states=True)
    hidden_state = outputs.hidden_states[TARGET_COMM_LAYER]
    mask = inputs.attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
    semantic_vector = torch.sum(hidden_state * mask, dim=1) / mask.sum(dim=1)
    return semantic_vector

@torch.no_grad()
def get_simulated_thought_sequence(prompt: str, max_new_tokens: int = 75) -> torch.Tensor:
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
    generated_ids = base_model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False
    )
    outputs = base_model(generated_ids, output_hidden_states=True)
    final_hidden_sequence = outputs.hidden_states[-1]
    return final_hidden_sequence

# --- 主缓存逻辑 ---
def run_caching_pipeline():
    print("🚀 Starting data caching pipeline...")
    
    data_files = {
        "analyst_to_trader": ("data/analyst_to_trader_scenarios.jsonl", "analyst_input", "ideal_trader_starting_thought"),
        "bull_to_bear": ("data/debate_scenarios.jsonl", "attacker_argument", "ideal_rebuttal_thought"),
        "bear_to_bull": ("data/debate_scenarios.jsonl", "rebuttal_argument", "ideal_counter_attack_thought"),
        "trader_to_risk": ("data/risk_scenarios.jsonl", "trader_plan_text", "ideal_risk_manager_thought"),
    }
    
    cache_dir = "data/cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    for name, (file_path, source_key, target_key) in data_files.items():
        print(f"\n--- Caching for: {name} ---")
        try:
            with open(file_path, "r") as f:
                scenarios = [json.loads(line) for line in f]
        except FileNotFoundError:
            print(f"WARNING: '{file_path}' not found. Skipping.")
            continue

        cached_data = []
        for i, scenario in enumerate(tqdm(scenarios, desc=f"Caching {name}")):
            source_text = scenario.get(source_key)
            target_text = scenario.get(target_key)
            if not source_text or not target_text: continue
            
            # 执行昂贵的计算
            source_agent_prompt = f"You are a financial analyst. Your task is to process the following information and form a conclusion. Information: '{source_text}'"
            h_source_sequence = get_simulated_thought_sequence(source_agent_prompt)
            h_target_mean_vector = get_semantic_vector(target_text)
            
            # 将张量保存到文件，并在列表中记录路径
            source_path = os.path.join(cache_dir, f"{name}_{i}_source.pt")
            target_path = os.path.join(cache_dir, f"{name}_{i}_target.pt")
            torch.save(h_source_sequence.cpu(), source_path)
            torch.save(h_target_mean_vector.cpu(), target_path)
            
            cached_data.append({"source_path": source_path, "target_path": target_path})
            
        # 将缓存文件的路径列表保存为一个新的jsonl文件
        cache_index_file = os.path.join(cache_dir, f"{name}_cached_index.jsonl")
        with open(cache_index_file, "w") as f:
            for item in cached_data:
                f.write(json.dumps(item) + "\n")
        print(f"✅ Finished caching for {name}. Index file created at '{cache_index_file}'.")

if __name__ == "__main__":
    run_caching_pipeline()