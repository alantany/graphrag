def get_openai_client():
    """获取统一配置的OpenAI客户端"""
    client = OpenAI(
        api_key=st.secrets["openrouter"]["api_key"],
        base_url=st.secrets["openrouter"]["base_url"]
    )
    
    # 添加OpenRouter所需的请求头
    client._default_headers = {
        "HTTP-Referer": "https://your-app-url.com",  # 替换为您的应用URL
        "X-Title": "AI知识问答系统"  # 您的应用名称
    }
    
    return client

def get_model_name():
    """获取当前配置的模型名称"""
    return st.secrets["openrouter"]["model"] 