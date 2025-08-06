# helios/graph/state.py

from typing import TypedDict, List, Optional, Dict
import torch

class HSC_AgentState(TypedDict):
    """
    定义在Helios工作流中传递的状态对象。
    它携带了驱动整个协作流程所需的所有信息。
    """
    # === 宏观追踪信息 ===
    company_of_interest: str
    trade_date: str
    
    # 用于记录和调试的文本日志，保持系统的可解释性
    text_log: List[str]

    # === 微观核心：隐藏状态 ===
    # 这是在Agent之间流动的主要信息载体，一个高维张量
    current_hidden_state: torch.Tensor
    
    # 分析师层会产生多个隐藏状态，用字典暂存
    # key是分析师角色(e.g., "fundamental"), value是其输出的隐藏状态
    analyst_hidden_states: Dict[str, torch.Tensor]

    # === 新增：辩论控制字段 ===
    debate_round: int  # 当前辩论轮次
    max_debate_rounds: int # 最大辩论轮次

    # === 最终决策存储 ===
    final_decision: Optional[str]