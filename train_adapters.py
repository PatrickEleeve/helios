# helios/train_adapters.py (最终完整版)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
import os
from typing import Dict, List, Any

from adapters.adapter import SemanticAdapter
from configs.config import BASE_MODEL_NAME, HIDDEN_SIZE, BOTTLENECK_DIM, TARGET_COMM_LAYER, DEVICE

# --- 1. 初始化模型、分词器 ---

print("Loading base model and tokenizer for training...")
# 为了性能，我们依然使用半精度，之前已经验证过速度提升巨大
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

# --- 2. 核心辅助函数 (重大修正) ---

@torch.no_grad()
def get_semantic_vector(text: str) -> torch.Tensor:
    """将【简洁】的目标文本编码成其核心语义【单一平均向量】。"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
    outputs = base_model(**inputs, output_hidden_states=True)
    hidden_state = outputs.hidden_states[TARGET_COMM_LAYER]
    mask = inputs.attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
    semantic_vector = torch.sum(hidden_state * mask, dim=1) / mask.sum(dim=1)
    return semantic_vector

@torch.no_grad()
def get_simulated_thought_sequence(prompt: str, max_new_tokens: int = 75) -> torch.Tensor:
    """
    通过模拟一个Agent完整的思考过程，获取其【最后一层完整的隐藏状态序列】。
    这是最符合模型工作原理的“思想”表示。
    """
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
    # 步骤1: 仅用generate获取生成的token ids
    generated_ids = base_model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False
    )
    # 步骤2: 将输入和输出的ids拼接，得到完整的序列
    full_sequence_ids = torch.cat([inputs.input_ids, generated_ids[:, inputs.input_ids.shape[1]:]], dim=1)
    
    # 步骤3: 对完整序列进行一次前向传播，以获取最干净、最完整的隐藏状态
    outputs = base_model(full_sequence_ids, output_hidden_states=True)
    # 返回最后一层的完整隐藏状态序列，形状为 [batch_size, seq_len, hidden_size]
    final_hidden_sequence = outputs.hidden_states[-1]
    return final_hidden_sequence

# --- 3. 核心训练逻辑封装 (重大修正) ---

def train_single_adapter(
    adapter: SemanticAdapter, 
    optimizer: optim.Optimizer,
    scheduler: CosineAnnealingLR,
    scenarios: List[Dict[str, Any]], 
    source_key: str, 
    target_key: str, 
    epochs: int,
    adapter_name: str
):
    """一个通用的函数，用于训练单个适配器。"""
    print(f"\n--- Training Adapter: {adapter_name} ({source_key} -> {target_key}) ---")
    adapter.train()
    cosine_loss_fn = nn.CosineEmbeddingLoss()
    epoch_progress_bar = tqdm(range(epochs), desc=f"Adapter: {adapter_name}")
    last_avg_loss = 0.0

    for epoch in epoch_progress_bar:
        total_loss = 0
        for scenario in scenarios:
            optimizer.zero_grad()
            source_text = scenario.get(source_key)
            target_text = scenario.get(target_key)
            if not source_text or not target_text: continue

            source_agent_prompt = f"You are a financial analyst. Your task is to process the following information and form a conclusion. Information: '{source_text}'"
            
            # --- 核心修改 ---
            # 1. 源“思想”现在是一个完整的序列
            h_source_sequence = get_simulated_thought_sequence(source_agent_prompt)
            
            # 2. 适配器翻译整个序列
            h_predicted_sequence = adapter(h_source_sequence)
            
            # 3. 我们希望翻译后的序列，其【平均思想】与我们的目标一致
            h_predicted_mean_vector = h_predicted_sequence.mean(dim=1)
            
            # 4. 目标思想依然是一个简洁的单一向量
            h_target_mean_vector = get_semantic_vector(target_text)
            
            y = torch.ones(h_source_sequence.shape[0]).to(DEVICE)
            loss = cosine_loss_fn(h_predicted_mean_vector, h_target_mean_vector, y)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        last_avg_loss = total_loss / len(scenarios) if scenarios else 0
        epoch_progress_bar.set_postfix(avg_loss=f"{last_avg_loss:.6f}", lr=f"{scheduler.get_last_lr()[0]:.1e}")
    
    print(f"Adapter '{adapter_name}' training finished. Final Avg Loss: {last_avg_loss:.6f}")

# --- 4. 主训练协调函数 (无需修改) ---
def run_training_pipeline():
    # ... (此函数内容完全不变，直接复用即可)
    print("🚀 Initializing training pipeline...")

    adapters_to_train = {
        "analyst_to_trader": SemanticAdapter(HIDDEN_SIZE, BOTTLENECK_DIM).to(DEVICE),
        "bull_to_bear": SemanticAdapter(HIDDEN_SIZE, BOTTLENECK_DIM).to(DEVICE),
        "bear_to_bull": SemanticAdapter(HIDDEN_SIZE, BOTTLENECK_DIM).to(DEVICE),
        "trader_to_risk": SemanticAdapter(HIDDEN_SIZE, BOTTLENECK_DIM).to(DEVICE)
    }
    print(f"Adapters to train: {list(adapters_to_train.keys())}")

    trainable_params = [p for adapter in adapters_to_train.values() for p in adapter.parameters()]
    optimizer = optim.AdamW(trainable_params, lr=5e-4, weight_decay=0.01)
    
    total_epochs = 80 
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)

    # 任务1
    try:
        with open("data/analyst_to_trader_scenarios.jsonl", "r") as f:
            scenarios_at = [json.loads(line) for line in f]
        train_single_adapter(
            adapters_to_train['analyst_to_trader'], 
            optimizer, scheduler, scenarios_at, 
            source_key='analyst_input', 
            target_key='ideal_trader_starting_thought',
            epochs=20, adapter_name="analyst_to_trader"
        )
    except FileNotFoundError:
        print("WARNING: 'data/analyst_to_trader_scenarios.jsonl' not found. Skipping.")

    # 任务2
    try:
        with open("data/debate_scenarios.jsonl", "r") as f:
            scenarios_db = [json.loads(line) for line in f]
        train_single_adapter(
            adapters_to_train['bull_to_bear'],
            optimizer, scheduler, scenarios_db,
            source_key='attacker_argument',
            target_key='ideal_rebuttal_thought',
            epochs=20, adapter_name="bull_to_bear"
        )
    except FileNotFoundError:
        print("WARNING: 'data/debate_scenarios.jsonl' not found. Skipping.")
        
    # 任务3
    try:
        with open("data/debate_scenarios_rebuttal.jsonl", "r") as f:
             scenarios_db_rebuttal = [json.loads(line) for line in f]
        train_single_adapter(
            adapters_to_train['bear_to_bull'],
            optimizer, scheduler, scenarios_db_rebuttal,
            source_key='rebuttal_argument',
            target_key='ideal_counter_attack_thought',
            epochs=20, adapter_name="bear_to_bull"
        )
    except FileNotFoundError:
        print("WARNING: 'data/debate_scenarios_rebuttal.jsonl' not found. Skipping.")

    # 任务4
    try:
        with open("data/trader_to_risk_scenarios.jsonl", "r") as f:
             scenarios_tr = [json.loads(line) for line in f]
        train_single_adapter(
            adapters_to_train['trader_to_risk'],
            optimizer, scheduler, scenarios_tr,
            source_key='trader_plan_text',
            target_key='ideal_risk_manager_thought',
            epochs=20, adapter_name="trader_to_risk"
        )
    except FileNotFoundError:
        print("WARNING: 'data/trader_to_risk_scenarios.jsonl' not found. Skipping.")

    print("\n✅ All training tasks finished. Saving adapter weights...")
    output_dir = "adapters"
    os.makedirs(output_dir, exist_ok=True)
    
    for name, adapter in adapters_to_train.items():
        if any(p.grad is not None for p in adapter.parameters()):
            save_path = os.path.join(output_dir, f"{name}_adapter.pth")
            torch.save(adapter.state_dict(), save_path)
            print(f"  -> Saved '{name}' adapter to '{save_path}'")
            
    print("Adapters saved successfully!")


if __name__ == "__main__":
    run_training_pipeline()