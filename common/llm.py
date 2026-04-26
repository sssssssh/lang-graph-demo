"""统一的 LLM 客户端工厂。

走 OpenAI 兼容协议，默认指向火山引擎方舟。
切换其他厂商（DeepSeek / 通义 / 智谱）只需改 .env，代码不动。

⚠️ 副作用提示：本模块在 import 时会调用 `load_dotenv()`，把仓库根 `.env` 加载到 `os.environ`。
   这意味着 `from common.llm import get_llm` 这一行就会读文件、改环境变量。
   若你写测试时希望完全隔离，请在 fixture 中用 `monkeypatch.setattr(os.environ, ...)`
   或 `patch.dict(os.environ, {...}, clear=True)`，并注意 `load_dotenv()` 默认 `override=False`，
   不会覆盖已存在的 env 变量。
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
