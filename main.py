# helios/main.py (最终完整版)

import torch
from langgraph.graph import StateGraph, END
from typing import Dict

from graph.state import HSC_AgentState
from configs.config import BASE_MODEL_NAME, HIDDEN_SIZE, BOTTLENECK_DIM, DEVICE,TARGET_COMM_LAYER
from adapters.adapter import SemanticAdapter
from transformers import AutoTokenizer, AutoModelForCausalLM

@torch.no_grad()
def agent_thinks(
    prompt: str,
    context_hidden_sequence: torch.Tensor,
    max_new_tokens: int = 75
) -> (str, torch.Tensor):
    """最终版：使用完整的隐藏状态序列作为“软提示”前缀。"""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    inputs_embeds = base_model.model.embed_tokens(inputs.input_ids)
    
    # 核心修改: 拼接上下文序列和词嵌入序列
    fused_embeds = torch.cat([context_hidden_sequence, inputs_embeds], dim=1)
    new_attention_mask = torch.ones(fused_embeds.shape[:2], device=DEVICE)

    # 步骤1: 仅用generate获取生成的token ids
    generated_ids = base_model.generate(
        inputs_embeds=fused_embeds,
        attention_mask=new_attention_mask,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    # 步骤2: 解码文本
    input_length = fused_embeds.shape[1] 
    newly_generated_ids = generated_ids[:, input_length:]
    generated_text = tokenizer.decode(newly_generated_ids[0], skip_special_tokens=True)

    # 步骤3: 对【包含了新生成内容】的完整序列进行一次前向传播，以获取其最终的隐藏状态序列
    full_sequence_ids = torch.cat([torch.zeros_like(context_hidden_sequence[:,:,0]).long(), inputs.input_ids, newly_generated_ids], dim=1) # 用0来表示虚拟token
    full_outputs = base_model(generated_ids, output_hidden_states=True)
    new_hidden_sequence = full_outputs.hidden_states[-1]

    return generated_text.strip(), new_hidden_sequence

# --- 1. 全局资源加载 (无需修改) ---
# ... (此部分代码完全不变)
print("Loading base model for inference...")
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=dtype
).to(DEVICE)
base_model.eval()
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
adapters = load_all_adapters() # load_all_adapters函数也无需修改

# --- 2. 定义Agent节点 (重大修正) ---

def fundamental_analyst_node(state: HSC_AgentState) -> HSC_AgentState:
    """基础面分析师节点，输出【序列】"""
    log_message = "Fundamental Analyst Node: Analyzing company fundamentals..."
    print(log_message); state['text_log'].append(log_message)
    input_text = "Company XYZ reported record profits and strong future guidance."
    with torch.no_grad():
        inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)
        outputs = base_model(**inputs, output_hidden_states=True)
        # 返回最后一层的完整序列
        h_fundamental_seq = outputs.hidden_states[-1]
    state['analyst_hidden_states']['fundamental'] = h_fundamental_seq
    return state

def sentiment_analyst_node(state: HSC_AgentState) -> HSC_AgentState:
    """情绪分析师节点，输出【序列】"""
    log_message = "Sentiment Analyst Node: Analyzing social media sentiment..."
    print(log_message); state['text_log'].append(log_message)
    input_text = "Market sentiment is fearful, with many traders expecting a downturn."
    with torch.no_grad():
        inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)
        outputs = base_model(**inputs, output_hidden_states=True)
        h_sentiment_seq = outputs.hidden_states[-1]
    state['analyst_hidden_states']['sentiment'] = h_sentiment_seq
    return state

def fusion_node(state: HSC_AgentState) -> HSC_AgentState:
    """融合节点，通过【拼接】合并序列"""
    log_message = "Fusion Node: Combining insights from all analysts..."
    print(log_message); state['text_log'].append(log_message)
    h_fundamental_seq = state['analyst_hidden_states']['fundamental']
    h_sentiment_seq = state['analyst_hidden_states']['sentiment']
    # 融合策略：拼接序列
    h_fused_seq = torch.cat([h_fundamental_seq, h_sentiment_seq], dim=1)
    state['current_hidden_state'] = h_fused_seq
    return state

# Agent节点（bull, bear, trader）现在接收和输出的都是序列
def bull_debater_node(state: HSC_AgentState) -> HSC_AgentState:
    state['debate_round'] += 1
    log_message = f"\n--- Debate Round {state['debate_round']} ---\nBull Debater: Formulating bullish argument from context..."
    print(log_message); state['text_log'].append(log_message)
    prompt = "You are a confident bullish financial debater. Based on the provided context, formulate a strong, concise bullish argument."
    generated_argument, new_hidden_sequence = agent_thinks(prompt, context_hidden_sequence=state['current_hidden_state'])
    print(f"  -> Bull's Argument: \"{generated_argument}\"")
    state['text_log'].append(f"Bull's Argument: {generated_argument}")
    state['current_hidden_state'] = new_hidden_sequence
    return state

def bear_debater_node(state: HSC_AgentState) -> HSC_AgentState:
    log_message = "Bear Debater: Analyzing context and formulating rebuttal..."
    print(log_message); state['text_log'].append(log_message)
    prompt = "You are a skeptical bearish financial debater. Analyze the provided context, find a flaw, and provide a sharp, concise rebuttal."
    generated_rebuttal, new_hidden_sequence = agent_thinks(prompt, context_hidden_sequence=state['current_hidden_state'])
    print(f"  -> Bear's Rebuttal: \"{generated_rebuttal}\"")
    state['text_log'].append(f"Bear's Rebuttal: {generated_rebuttal}")
    state['current_hidden_state'] = new_hidden_sequence
    return state

def trader_node(state: HSC_AgentState) -> HSC_AgentState:
    log_message = "\n--- Decision Layer ---\nTrader Node: Analyzing final debate context to form a plan..."
    print(log_message); state['text_log'].append(log_message)
    prompt = "You are a decisive trader. Based on the entire debate context, make a clear trading decision. Your final answer must be one of: BUY, SELL, or HOLD."
    generated_decision, new_hidden_sequence = agent_thinks(prompt, context_hidden_sequence=state['current_hidden_state'], max_new_tokens=5)
    decision = "HOLD"
    if "BUY" in generated_decision.upper(): decision = "BUY"
    elif "SELL" in generated_decision.upper(): decision = "SELL"
    print(f"  -> Trader's Plan: \"{decision}\"")
    state['final_decision'] = decision
    state['text_log'].append(f"Trader's initial plan: {decision}")
    state['current_hidden_state'] = new_hidden_sequence
    return state

def risk_manager_node(state: HSC_AgentState) -> HSC_AgentState:
    """风险经理节点，基于【序列平均值】进行评估"""
    log_message = "Risk Manager: Evaluating the proposed trade plan..."
    print(log_message); state['text_log'].append(log_message)
    # 先对序列取平均，再计算范数
    risk_score = torch.norm(state['current_hidden_state'].mean(dim=1)).item()
    if risk_score > 10:
        state['text_log'].append("Risk Assessment: High risk detected (high conviction/volatility).")
    else:
        state['text_log'].append("Risk Assessment: Risk level is acceptable.")
    return state

# portfolio_manager_node 无需修改

# --- 3. 定义“神经边”与条件 (无需修改) ---
# ...

# --- 4. 构建图 (无需修改) ---
# ...

# --- 5. 运行图 (重大修正) ---

if __name__ == "__main__":
    print("🚀 Starting Helios Final Workflow...")

    # 初始状态现在是一个长度为1的序列，代表一个“起始思考”token
    initial_hidden_state_seq = torch.zeros(1, 1, HIDDEN_SIZE).to(DEVICE)

    initial_state = HSC_AgentState(
        company_of_interest="NVDA",
        trade_date="2025-08-08",
        text_log=[],
        current_hidden_state=initial_hidden_state_seq, # <-- 使用序列进行初始化
        analyst_hidden_states={},
        final_decision=None,
        debate_round=0,
        max_debate_rounds=2
    )

    final_state = app.invoke(initial_state)

    print("\n--- Workflow Finished ---")
    print(f"Final Decision for {final_state['company_of_interest']} on {final_state['trade_date']}: {final_state['final_decision']}")
    print("\n--- Full Execution Log ---")
    for i, log in enumerate(final_state['text_log']):
        print(f"{i+1:02d}: {log}")