# helios/train_adapters.py (最终完整、可运行的Seq2Seq训练版)

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
import os
from typing import Dict, List, Any

from adapters.adapter import SemanticAdapter
from configs.config import BASE_MODEL_NAME, HIDDEN_SIZE, BOTTLENECK_DIM, TARGET_COMM_LAYER, DEVICE

# --- 1. 全局资源加载 ---
print("Loading base model and tokenizer...")
# 使用与main.py相同的dtype设置，确保一致性
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=dtype
).to(DEVICE)
base_model.eval()
print("Freezing base model parameters...")
for param in base_model.parameters():
    param.requires_grad = False

# --- 2. 核心辅助函数 ---
@torch.no_grad()
def get_simulated_thought_sequence(prompt: str, max_new_tokens: int = 75) -> torch.Tensor:
    """
    通过模拟一个Agent完整的思考过程，获取其【最后一层完整的隐藏状态序列】。
    """
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
    # 使用 .generate() 来创建包含思考过程的完整token序列
    generated_ids = base_model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False  # 在生成思想时使用确定性生成
    )
    # 对完整的生成序列进行一次前向传播，以获取最准确的隐藏状态
    outputs = base_model(generated_ids, output_hidden_states=True)
    final_hidden_sequence = outputs.hidden_states[-1]
    return final_hidden_sequence

# --- 3. 核心训练逻辑 (序列到序列版) ---
def train_single_adapter_seq2seq(
    adapter: SemanticAdapter, 
    optimizer: optim.Optimizer,
    scenarios: List[Dict[str, str]], 
    epochs: int,
    adapter_name: str
):
    """
    一个通用的、从缓存加载数据来训练单个适配器的函数 (Seq2Seq模式)。
    """
    print(f"\n--- Training Adapter (Seq2Seq): {adapter_name} ---")
    adapter.train()
    # 使用均方误差损失来直接比较两个序列
    mse_loss_fn = nn.MSELoss()
    epoch_progress_bar = tqdm(range(epochs), desc=f"Adapter: {adapter_name}")
    
    for epoch in epoch_progress_bar:
        total_loss = 0
        for scenario in scenarios:
            optimizer.zero_grad()
            
            # 从磁盘加载预计算好的源序列和目标序列
            h_source_sequence = torch.load(scenario['source_path']).to(DEVICE)
            h_target_sequence = torch.load(scenario['target_path']).to(DEVICE)
            
            # 适配器翻译整个源序列
            h_predicted_sequence = adapter(h_source_sequence)
            
            # 对齐序列长度以便计算损失
            # 这是为了处理源和目标生成不同长度的情况
            len_target = h_target_sequence.shape[1]
            len_predicted = h_predicted_sequence.shape[1]
            
            min_len = min(len_target, len_predicted)
            
            h_predicted_aligned = h_predicted_sequence[:, :min_len, :]
            h_target_aligned = h_target_sequence[:, :min_len, :]

            # 计算两个对齐后的完整序列之间的损失
            loss = mse_loss_fn(h_predicted_aligned, h_target_aligned)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(scenarios) if scenarios else 0
        epoch_progress_bar.set_postfix(avg_loss=f"{avg_loss:.6f}")
    
    print(f"Adapter '{adapter_name}' training finished. Final Avg Loss: {avg_loss:.6f}")

# --- 4. 主程序入口 (整合缓存和训练) ---
if __name__ == "__main__":
    CACHE_DIR = "data/cache"
    DATA_DIR = "data"
    
    # 定义所有需要的数据和缓存文件
    data_files_info = {
        "analyst_to_trader": {"source_file": os.path.join(DATA_DIR, "analyst_to_trader_scenarios.jsonl"), "source_key": "analyst_input", "target_key": "ideal_trader_starting_thought"},
        "bull_to_bear":      {"source_file": os.path.join(DATA_DIR, "debate_scenarios.jsonl"), "source_key": "attacker_argument", "target_key": "ideal_rebuttal_thought"},
        "bear_to_bull":      {"source_file": os.path.join(DATA_DIR, "debate_scenarios_rebuttal.jsonl"), "source_key": "rebuttal_argument", "target_key": "ideal_counter_attack_thought"},
        "trader_to_risk":    {"source_file": os.path.join(DATA_DIR, "trader_to_risk_scenarios.jsonl"), "source_key": "trader_plan_text", "target_key": "ideal_risk_manager_thought"},
    }

    # --- 步骤 1: 自动检查并生成缓存 ---
    print("--- Checking and Generating Data Cache (Seq2Seq Mode) ---")
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    for name, info in data_files_info.items():
        cache_index_file = os.path.join(CACHE_DIR, f"{name}_cached_index_s2s.jsonl")
        if os.path.exists(cache_index_file):
            print(f"Seq2Seq cache for '{name}' already exists. Skipping generation.")
            continue
        
        print(f"\n--- Caching for (Seq2Seq): {name} ---")
        try:
            with open(info["source_file"], "r") as f:
                scenarios = [json.loads(line) for line in f]
        except FileNotFoundError:
            print(f"WARNING: Source file '{info['source_file']}' not found. Cannot generate cache.")
            continue

        cached_data = []
        for i, scenario in enumerate(tqdm(scenarios, desc=f"Caching {name}")):
            source_text = scenario.get(info["source_key"])
            target_text = scenario.get(info["target_key"])
            if not source_text or not target_text: continue
            
            # 为源和目标都创建合适的prompt
            source_prompt = f"As a financial expert, analyze the following information and form a conclusion: '{source_text}'"
            target_prompt = f"As a financial expert, your thought process should now shift to this: '{target_text}'"
            
            # 源和目标现在都是完整的思想序列
            h_source_sequence = get_simulated_thought_sequence(source_prompt)
            h_target_sequence = get_simulated_thought_sequence(target_prompt)
            
            source_path = os.path.join(CACHE_DIR, f"{name}_{i}_source_s2s.pt")
            target_path = os.path.join(CACHE_DIR, f"{name}_{i}_target_s2s.pt")
            torch.save(h_source_sequence.cpu(), source_path)
            torch.save(h_target_sequence.cpu(), target_path)
            
            cached_data.append({"source_path": source_path, "target_path": target_path})
            
        with open(cache_index_file, "w") as f:
            for item in cached_data:
                f.write(json.dumps(item) + "\n")
        print(f"✅ Finished Seq2Seq caching for {name}. Index file created at '{cache_index_file}'.")

    # --- 步骤 2: 开始Seq2Seq训练 ---
    print("\n🚀 Initializing Seq2Seq training pipeline from cache...")
    model_dtype = base_model.dtype
    adapters = {name: SemanticAdapter(HIDDEN_SIZE, BOTTLENECK_DIM).to(device=DEVICE, dtype=model_dtype) for name in data_files_info.keys()}
    optimizer = optim.AdamW([p for a in adapters.values() for p in a.parameters()], lr=5e-4, weight_decay=0.01)

    for name, info in data_files_info.items():
        cache_index_file = os.path.join(CACHE_DIR, f"{name}_cached_index_s2s.jsonl")
        try:
            with open(cache_index_file, "r") as f:
                scenarios = [json.loads(line) for line in f]
            # 调用新的序列到序列训练函数
            train_single_adapter_seq2seq(
                adapters[name], optimizer, scenarios, 
                epochs=20, adapter_name=name
            )
        except FileNotFoundError:
            print(f"WARNING: Cache index for '{name}' not found. Skipping training. Please run caching first.")

    # --- 步骤 3: 保存适配器 ---
    print("\n✅ All training tasks finished. Saving adapter weights...")
    output_dir = "adapters"
    os.makedirs(output_dir, exist_ok=True)
    for name, adapter in adapters.items():
        save_path = os.path.join(output_dir, f"{name}_adapter.pth")
        torch.save(adapter.state_dict(), save_path)
        print(f"  -> Saved '{name}' adapter to '{save_path}'")
        
    print("\n🎉 Training pipeline complete! Your adapters are ready. 🎉")