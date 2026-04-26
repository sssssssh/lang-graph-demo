"""测试 get_llm()：从 env 读取配置，缺失时清晰报错。"""
import os
import pytest
from unittest.mock import patch

from common.llm import get_llm


def test_get_llm_raises_when_api_key_missing():
    """ARK_API_KEY 没设置时，应当 raise，且 message 中包含变量名提示用户。"""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(RuntimeError, match="ARK_API_KEY"):
            get_llm()


def test_get_llm_raises_when_model_missing():
    """LLM_MODEL 没设置时，应当 raise，且 message 中包含变量名。"""
    with patch.dict(os.environ, {"ARK_API_KEY": "fake"}, clear=True):
        with pytest.raises(RuntimeError, match="LLM_MODEL"):
            get_llm()


def test_get_llm_returns_chat_openai_when_env_present():
    """env 齐全时返回 ChatOpenAI 实例，且 base_url / model / api_key 配置正确。"""
    from langchain_openai import ChatOpenAI

    env = {
        "ARK_API_KEY": "fake-key",
        "LLM_MODEL": "doubao-test",
        "ARK_BASE_URL": "https://ark.example.com/v3",
    }
    with patch.dict(os.environ, env, clear=True):
        llm = get_llm(temperature=0.5)
    # 用 model_dump() 跨 LangChain 版本稳定地检查配置（不依赖具体属性名）
    assert isinstance(llm, ChatOpenAI)
    assert llm.temperature == 0.5
    dump_str = str(llm.model_dump())
    assert "doubao-test" in dump_str
    assert "ark.example.com" in dump_str


def test_get_llm_uses_default_base_url_when_not_set():
    """ARK_BASE_URL 未设置时，应当走默认的火山方舟 endpoint。"""
    env = {"ARK_API_KEY": "fake-key", "LLM_MODEL": "doubao-test"}
    with patch.dict(os.environ, env, clear=True):
        llm = get_llm()
    assert "ark.cn-beijing.volces.com" in str(llm.model_dump())
