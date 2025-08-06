# helios/main.py (最终架构：返璞归真版)

import torch
from langgraph.graph import StateGraph, END
from typing import Dict, List

from graph.state import HSC_AgentState
from configs.config import BASE_MODEL_NAME, HIDDEN_SIZE, BOTTLENECK_DIM, DEVICE
from adapters.adapter import SemanticAdapter
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 1. 全局资源加载 ---
print("Loading base model for inference...")
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=dtype).to(DEVICE)
base_model.eval()
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

# --- 2. 核心思考引擎 (彻底简化) ---
@torch.no_grad()
def agent_thinks(prompt: str, max_new_tokens: int = 200) -> (str, torch.Tensor):
    """
    最终简化版：只接收一个完整的文本prompt，输出生成的文本和最终的隐藏状态序列。
    """
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(DEVICE)
    
    generated_ids = base_model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    
    generated_text = tokenizer.decode(generated_ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # 对完整的、包含了新生成内容的序列进行一次前向传播，以获取其最终的隐藏状态
    full_outputs = base_model(generated_ids, output_hidden_states=True)
    new_hidden_sequence = full_outputs.hidden_states[-1]

    return generated_text.strip(), new_hidden_sequence

# --- 3. 适配器加载 (无需修改) ---
def load_all_adapters() -> Dict[str, SemanticAdapter]:
    # ... (此部分代码完全不变)
    print("Loading all trained adapters...")
    adapter_paths = {
        "analyst_to_trader": "adapters/analyst_to_trader_adapter.pth",
        "bull_to_bear": "adapters/bull_to_bear_adapter.pth",
        "bear_to_bull": "adapters/bear_to_bull_adapter.pth",
        "trader_to_risk": "adapters/trader_to_risk_adapter.pth"
    }
    loaded_adapters = {}
    model_dtype = base_model.dtype
    for name, path in adapter_paths.items():
        try:
            adapter = SemanticAdapter(HIDDEN_SIZE, BOTTLENECK_DIM).to(device=DEVICE, dtype=model_dtype)
            adapter.load_state_dict(torch.load(path, map_location=DEVICE))
            adapter.eval()
            loaded_adapters[name] = adapter
            print(f"  - Adapter '{name}' loaded successfully.")
        except FileNotFoundError:
            print(f"  - WARNING: Adapter file not found at '{path}'.")
            loaded_adapters[name] = None
        except Exception as e:
            print(f"  - ERROR: Failed to load adapter '{name}'. Reason: {e}")
            loaded_adapters[name] = None
    return loaded_adapters

adapters = load_all_adapters()

# --- 4. 定义Agent节点 (全新逻辑) ---

def analyst_node(state: HSC_AgentState) -> HSC_AgentState:
    """统一的分析师节点，一次性生成所有分析报告。"""
    log_message = "Analyst Team: Generating all reports..."
    print(log_message); state['text_log'].append(log_message)
    
    prompt = (
        "You are an expert financial analyst team. Provide a concise summary for each of the following areas:\n"
        "1. Fundamental Analysis: Based on Company XYZ reporting record profits and strong future guidance.\n"
        "2. Sentiment Analysis: Based on the market sentiment being fearful, with many traders expecting a downturn.\n\n"
        "Provide your reports."
    )
    
    # 分析师团队的思考是整个流程的起点
    generated_reports, new_hidden_state = agent_thinks(prompt, max_new_tokens=200)
    
    state['current_hidden_state'] = new_hidden_state
    state['text_log'].append(f"  -> Analyst Team Reports:\n{generated_reports}")
    return state

def bull_debater_node(state: HSC_AgentState) -> HSC_AgentState:
    """辩手节点现在基于完整的文本历史进行思考。"""
    state['debate_round'] += 1
    log_message = f"\n--- Debate Round {state['debate_round']} ---\nBull Debater: Formulating argument..."
    print(log_message); state['text_log'].append(log_message)
    
    # 构建包含所有历史的完整prompt
    full_history = "\n".join(state['text_log'])
    prompt = (
        f"You are a confident bullish debater. Here is the conversation so far:\n\n"
        f"--- START OF HISTORY ---\n{full_history}\n--- END OF HISTORY ---\n\n"
        f"Based on all the information above, present your strongest bullish argument now:"
    )
    
    generated_argument, new_hidden_state = agent_thinks(prompt)
    
    print(f"  -> Bull's Argument: \"{generated_argument}\"")
    state['text_log'].append(f"Bull's Argument: {generated_argument}")
    state['current_hidden_state'] = new_hidden_state
    return state

# Bear, Trader, Risk Manager 节点都遵循与 Bull Debater 类似的逻辑
def bear_debater_node(state: HSC_AgentState) -> HSC_AgentState:
    log_message = "Bear Debater: Formulating rebuttal..."
    print(log_message); state['text_log'].append(log_message)
    full_history = "\n".join(state['text_log'])
    
    # 使用更直接、更具命令性的Prompt
    prompt = (
        f"You are a skeptical bearish debater. Below is the history of a financial debate. Your task is to provide a direct rebuttal to the last statement made by the Bull Debater.\n\n"
        f"--- DEBATE HISTORY ---\n{full_history}\n--- END OF HISTORY ---\n\n"
        f"Directly state your rebuttal now. Do not explain your steps. Do not use markdown. Start your response with 'The bull's argument is flawed because...':"
    )
    
    generated_rebuttal, new_hidden_state = agent_thinks(prompt)
    print(f"  -> Bear's Rebuttal: \"{generated_rebuttal}\"")
    state['text_log'].append(f"Bear's Rebuttal: {generated_rebuttal}")
    state['current_hidden_state'] = new_hidden_state
    return state

def trader_node(state: HSC_AgentState) -> HSC_AgentState:
    log_message = "\n--- Decision Layer ---\nTrader Node: Making a decision..."
    print(log_message); state['text_log'].append(log_message)
    full_history = "\n".join(state['text_log'])
    prompt = (
        f"You are the Trader. After reviewing the entire debate, make a final decision. Your answer must be one word: BUY, SELL, or HOLD.\n\n"
        f"--- DEBATE HISTORY ---\n{full_history}\n--- END OF HISTORY ---\n\n"
        f"Your Decision:"
    )
    generated_decision, new_hidden_state = agent_thinks(prompt, max_new_tokens=200)
    decision = "HOLD"
    if "BUY" in generated_decision.upper(): decision = "BUY"
    elif "SELL" in generated_decision.upper(): decision = "SELL"
    print(f"  -> Trader's Plan: \"{decision}\"")
    state['final_decision'] = decision
    state['text_log'].append(f"Trader's Plan: {decision}")
    state['current_hidden_state'] = new_hidden_state
    return state

# ... 其他节点也类似地修改 ...
def risk_manager_node(state: HSC_AgentState) -> HSC_AgentState:
    log_message = "Risk Manager: Evaluating the trade plan..."
    print(log_message); state['text_log'].append(log_message)
    full_history = "\n".join(state['text_log'])
    prompt = f"You are the Risk Manager. The trader has proposed a plan. Briefly assess the risk.\n\n--- HISTORY ---\n{full_history}\n--- END ---\n\nYour Risk Assessment:"
    risk_assessment, _ = agent_thinks(prompt, max_new_tokens=200)
    state['text_log'].append(f"Risk Assessment: {risk_assessment}")
    if "HIGH" in risk_assessment.upper() or "SIGNIFICANT" in risk_assessment.upper():
        state['high_risk_flag'] = True
    return state

def portfolio_manager_node(state: HSC_AgentState) -> HSC_AgentState:
    log_message = "Portfolio Manager: Final approval..."
    print(log_message); state['text_log'].append(log_message)
    final_decision = state['final_decision']
    if state.get('high_risk_flag', False):
        final_decision = "REJECTED (HOLD)"
        state['text_log'].append(f"Final Verdict: Trade REJECTED due to high risk. Final position: HOLD")
    else:
        state['text_log'].append(f"Final Verdict: Trade APPROVED. Executing final decision: {final_decision}.")
    state['final_decision'] = final_decision
    return state

# --- 5. 定义边和图 (结构简化) ---
def create_adapter_edge(adapter_name: str, log_template: str):
    # ... (这个函数保持不变)
    def edge_func(state: HSC_AgentState) -> HSC_AgentState:
        log_message = log_template.format(adapter_name=adapter_name)
        print(log_message); state['text_log'].append(log_message)
        adapter = adapters.get(adapter_name)
        if adapter:
            with torch.no_grad():
                state['current_hidden_state'] = adapter(state['current_hidden_state'])
        else:
            state['text_log'].append(f"WARNING: Adapter '{adapter_name}' not loaded. Passing state directly.")
        return state
    return edge_func

debate_to_trader_edge = create_adapter_edge("analyst_to_trader", "Edge: Debate -> Trader (Translating final debate state to trader context)")
bull_to_bear_edge = create_adapter_edge("bull_to_bear", "Edge: Bull -> Bear (Translating argument to challenge)")
bear_to_bull_edge = create_adapter_edge("bear_to_bull", "Edge: Bear -> Bull (Translating rebuttal to counter-challenge)")
trader_to_risk_edge = create_adapter_edge("trader_to_risk", "Edge: Trader -> Risk Manager (Translating plan to risk context)")

def should_continue_debate(state: HSC_AgentState) -> str:
    # ... (这个函数保持不变)
    if state['debate_round'] < state['max_debate_rounds']:
        return "continue_debate"
    else:
        log_message = "\n--- Debate Finished ---"
        print(log_message); state['text_log'].append(log_message)
        return "end_debate"

workflow = StateGraph(HSC_AgentState)
workflow.add_node("analyst_team", analyst_node) # 简化为一个分析师团队节点
workflow.add_node("bull_debater", bull_debater_node)
workflow.add_node("bear_debater", bear_debater_node)
workflow.add_node("trader", trader_node)
workflow.add_node("risk_manager", risk_manager_node)
workflow.add_node("portfolio_manager", portfolio_manager_node)
workflow.add_node("debate_to_trader_adapter_node", debate_to_trader_edge)
workflow.add_node("bull_to_bear_adapter_node", bull_to_bear_edge)
workflow.add_node("bear_to_bull_adapter_node", bear_to_bull_edge)
workflow.add_node("trader_to_risk_adapter_node", trader_to_risk_edge)

workflow.set_entry_point("analyst_team")
workflow.add_edge("analyst_team", "bull_debater") # 分析结束后直接开始辩论
# ... (其余的图连接逻辑与之前相同)
workflow.add_conditional_edges(
    "bull_debater", should_continue_debate,
    {"continue_debate": "bull_to_bear_adapter_node", "end_debate": "debate_to_trader_adapter_node"}
)
workflow.add_edge("bull_to_bear_adapter_node", "bear_debater")
workflow.add_edge("bear_debater", "bear_to_bull_adapter_node")
workflow.add_edge("bear_to_bull_adapter_node", "bull_debater")
workflow.add_edge("debate_to_trader_adapter_node", "trader")
workflow.add_edge("trader", "trader_to_risk_adapter_node")
workflow.add_edge("trader_to_risk_adapter_node", "risk_manager")
workflow.add_edge("risk_manager", "portfolio_manager")
workflow.add_edge("portfolio_manager", END)

app = workflow.compile()

# --- 6. 运行图 ---
if __name__ == "__main__":
    print("🚀 Starting Helios Final Workflow (Simplified Prompting)...")
    
    initial_state = HSC_AgentState(
        company_of_interest="NVDA",
        trade_date="2025-08-08",
        text_log=[],
        # 初始隐藏状态不再重要，因为第一个节点会创建它
        current_hidden_state=None, 
        final_decision=None,
        debate_round=0,
        max_debate_rounds=2,
        high_risk_flag=False
    )

    final_state = app.invoke(initial_state)

    print("\n--- Workflow Finished ---")
    print(f"Final Decision for {final_state['company_of_interest']} on {final_state['trade_date']}: {final_state['final_decision']}")
    print("\n--- Full Execution Log ---")
    for i, log in enumerate(final_state['text_log']):
        print(f"{i+1:02d}: {log}")