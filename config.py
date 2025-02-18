import streamlit as st
from openai import OpenAI

def get_openai_client():
    """获取统一配置的DeepSeek客户端"""
    client = OpenAI(
        api_key=st.secrets["deepseek"]["api_key"],
        base_url=st.secrets["deepseek"]["base_url"]
    )
    
    # 添加OpenRouter所需的请求头
    client._default_headers = {
        "HTTP-Referer": "https://your-app-url.com",  # 替换为您的应用URL
        "X-Title": "AI知识问答系统"  # 您的应用名称
    }
    
    return client

def get_model_name():
    """获取当前配置的模型名称"""
    return st.secrets["deepseek"]["model"]

def validate_config():
    """验证配置是否正确"""
    required_secrets = {
        "deepseek": ["api_key", "base_url", "model"],
        "neo4j": ["uri", "username", "password"]
    }
    
    for section, keys in required_secrets.items():
        if section not in st.secrets:
            raise ValueError(f"缺少配置部分：{section}")
        for key in keys:
            if key not in st.secrets[section]:
                raise ValueError(f"在 {section} 中缺少配置项：{key}")
    
    return True 