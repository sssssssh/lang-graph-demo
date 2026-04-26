"""统一的 LLM 客户端工厂。

走 OpenAI 兼容协议，默认指向火山引擎方舟。
切换其他厂商（DeepSeek / 通义 / 智谱）只需改 .env，代码不动。
"""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 仓库根目录的 .env 自动加载（多次调用幂等）
load_dotenv()

DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"


def get_llm(temperature: float = 0.3, **kwargs) -> ChatOpenAI:
    """构造 ChatOpenAI 实例。

    必需 env：ARK_API_KEY、LLM_MODEL
    可选 env：ARK_BASE_URL（默认走火山方舟）

    其他 ChatOpenAI 参数通过 **kwargs 透传，比如 streaming=True。
    """
    api_key = os.environ.get("ARK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ARK_API_KEY 未设置。请把 .env.example 复制为 .env 并填入火山引擎 API Key。"
        )

    model = os.environ.get("LLM_MODEL")
    if not model:
        raise RuntimeError(
            "LLM_MODEL 未设置。请在 .env 中填入模型名，例如 doubao-1-5-pro-32k-250115。"
        )

    base_url = os.environ.get("ARK_BASE_URL", DEFAULT_BASE_URL)

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        **kwargs,
    )
