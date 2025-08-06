# helios/configs/config.py

import os
from dotenv import load_dotenv

load_dotenv() # 加载 .env 文件中的环境变量

# --- API Keys ---
# 即便使用本地模型，保留这个结构也方便未来扩展
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 

# --- 模型设置 ---
# 我们将从这里开始，使用一个较小的、可快速测试的模型
BASE_MODEL_NAME = "Qwen/Qwen2-0.5B"
# 基础模型的隐藏层维度，Qwen2-0.5B是896
HIDDEN_SIZE = 896
# 适配器的瓶颈层维度
BOTTLENECK_DIM = 128
# 我们选择在哪一层进行隐藏状态的交换
TARGET_COMM_LAYER = 12 

# --- 设备设置 ---
DEVICE = "mps" # "cuda" if torch.cuda.is_available() else "cpu"

TARGET_COMM_LAYER = 12