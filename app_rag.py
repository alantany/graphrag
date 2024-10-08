"""
AI知识问答系统安装文档


1. 环境要求：
   - Python 3.7+
   - pip (Python包管理器)

2. 创建虚拟环境（可选但推荐）：
   python -m venv venv
   source venv/bin/activate  # 在Windows上使用: venv\Scripts\activate

3. 安装依赖：
   pip install -r requirements.txt

4. requirements.txt 文件内容：
altair==5.4.1
annotated-types==0.7.0
anyio==4.4.0
attrs==24.2.0
blinker==1.8.2
cachetools==5.5.0
certifi==2024.8.30
charset-normalizer==3.3.2
click==8.1.7
distro==1.9.0
faiss-cpu==1.8.0.post1
filelock==3.16.1
Flask==3.0.3
fsspec==2024.9.0
gitdb==4.0.11
GitPython==3.1.43
google-api-core==2.19.2
google-api-python-client==2.146.0
google-auth==2.34.0
google-auth-httplib2==0.2.0
google_search_results==2.4.2
googleapis-common-protos==1.65.0
h11==0.14.0
httpcore==1.0.5
httplib2==0.22.0
httpx==0.27.2
huggingface-hub==0.25.0
idna==3.10
itsdangerous==2.2.0
jieba==0.42.1
Jinja2==3.1.4
jiter==0.5.0
joblib==1.4.2
jsonschema==4.23.0
jsonschema-specifications==2023.12.1
lxml==5.3.0
markdown-it-py==3.0.0
MarkupSafe==2.1.5
mdurl==0.1.2
mpmath==1.3.0
narwhals==1.8.1
networkx==3.3
numpy==1.26.4
openai==1.46.0
packaging==24.1
pandas==2.2.2
pillow==10.4.0
proto-plus==1.24.0
protobuf==5.28.1
pyarrow==17.0.0
pyasn1==0.6.1
pyasn1_modules==0.4.1
PyAudio==0.2.11
pydantic==2.9.2
pydantic_core==2.23.4
pydeck==0.9.1
Pygments==2.18.0
pyparsing==3.1.4
PyPDF2==3.0.1
python-dateutil==2.9.0.post0
python-docx==1.1.2
pytz==2024.2
PyYAML==6.0.2
referencing==0.35.1
regex==2024.9.11
requests==2.32.3
rich==13.8.1
rpds-py==0.20.0
rsa==4.9
safetensors==0.4.5
scikit-learn==1.5.2
scipy==1.14.1
sentence-transformers==3.1.0
six==1.16.0
smmap==5.0.1
sniffio==1.3.1
SpeechRecognition==3.10.4
streamlit==1.38.0
sympy==1.13.2
tenacity==8.5.0
threadpoolctl==3.5.0
tiktoken==0.7.0
tokenizers==0.19.1
toml==0.10.2
torch==2.2.2
tornado==6.4.1
tqdm==4.66.5
transformers==4.44.2
typing_extensions==4.12.2
tzdata==2024.1
uritemplate==4.1.1
urllib3==2.2.3
Werkzeug==3.0.4


5. 其他依赖：
   - 确保你有有效的OpenAI API密钥
   - 如果使用Google搜索功能，需要有效的SerpAPI密钥
   - 对于语音识别功能，需要安装系统级依赖 PortAudio。这在某些托管环境（如 Streamlit Cloud）中可能无法直接安装。
     如果在托管环境中遇到问题，可能需要考虑使用基于浏览器的语音识别方案或云端语音识别服务。
   - 对于大型文件处理，可能需要增加系统内存或使用更强大的硬件

6. 运行应用：
   streamlit run app.py

注意：
- 请确保所有依赖都已正确安装
- 在代码中替换OpenAI API密钥和SerpAPI密钥为你自己的密钥
- 对于语音识别功能，需要安装系统级依赖 PortAudio。这在某些托管环境（如 Streamlit Cloud）中可能无法直接安装。
  如果在托管环境中遇到问题，可能需要考虑使用基于浏览器的语音识别方案或云端语音识别服务。
- 对于大型文件处理，可能需要增加系统内存或使用更强大的硬件
"""

import streamlit as st

# 设置页面配置必须是第一个 Streamlit 命令
st.set_page_config(
    page_title="AI知识问答系统",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None
)

# 隐藏 Streamlit 默认的菜单、页脚和 Deploy 按钮
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

from openai import OpenAI
from sentence_transformers import SentenceTransformer
import multiprocessing
import PyPDF2
import docx
import faiss
import tiktoken
import os
import pickle
import numpy as np
import jieba
from collections import Counter
import sqlite3
import pandas as pd
from serpapi import GoogleSearch
import requests
import io
import speech_recognition as sr
import streamlit.components.v1 as components
import time

# 初始化
client = OpenAI(
    api_key="sk-1pUmQlsIkgla3CuvKTgCrzDZ3r0pBxO608YJvIHCN18lvOrn",
    base_url="https://api.chatanywhere.tech/v1"
)

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# 计算token数量
def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(string))

# 文档向量化模块
def vectorize_document(file, max_tokens):
    text = ""
    if file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    else:
        text = file.getvalue().decode("utf-8")
    
    chunks = []
    current_chunk = ""
    for sentence in text.split('.'):
        if num_tokens_from_string(current_chunk + sentence) > max_tokens:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
        else:
            current_chunk += sentence + '.'
    if current_chunk:
        chunks.append(current_chunk)
    
    vectors = model.encode(chunks)
    index = faiss.IndexFlatL2(384)  # 384是向量维度,根据实际模型调整
    index.add(vectors)
    return chunks, index

# 新增函数：提取关键词
def extract_keywords(text, top_k=5):
    words = jieba.cut(text)
    word_count = Counter(words)
    # 过滤掉停用词和单个字符
    keywords = [word for word, count in word_count.most_common(top_k*2) if len(word) > 1]
    return keywords[:top_k]

# 新增函数：基于关键词搜文档
def search_documents(keywords, file_indices):
    relevant_docs = []
    for file_name, (chunks, _) in file_indices.items():
        doc_content = ' '.join(chunks)
        if any(keyword in doc_content for keyword in keywords):
            relevant_docs.append(file_name)
    return relevant_docs

# 修改知识问答模块
def rag_qa(query, file_indices, relevant_docs=None):
    keywords = extract_keywords(query)
    if relevant_docs is None:
        relevant_docs = search_documents(keywords, file_indices)
    
    if not relevant_docs:
        return "没有找到相关文档。请尝试使用不同的关键词。", [], ""

    all_chunks = []
    chunk_to_file = {}
    combined_index = faiss.IndexFlatL2(384)
    
    offset = 0
    for file_name in relevant_docs:
        if file_name in file_indices:
            chunks, index = file_indices[file_name]
            all_chunks.extend(chunks)
            for i in range(len(chunks)):
                chunk_to_file[offset + i] = file_name
            combined_index.add(index.reconstruct_n(0, index.ntotal))
            offset += len(chunks)

    if not all_chunks:
        return "没有找到相关信息。请确保已上传文档。", [], ""

    query_vector = model.encode([query])
    D, I = combined_index.search(query_vector, k=3)
    context = []
    context_with_sources = []
    for i in I[0]:
        if 0 <= i < len(all_chunks):  # 确引在有效范围内
            chunk = all_chunks[i]
            context.append(chunk)
            file_name = chunk_to_file.get(i, "未知文件")
            context_with_sources.append((file_name, chunk))

    context_text = "\n".join(context)
    
    # 确保总token数不超过4096
    max_context_tokens = 3000  # 为系统消息、查询和其他内容预留更多空间
    while num_tokens_from_string(context_text) > max_context_tokens:
        context_text = context_text[:int(len(context_text)*0.9)]  # 每次减少10%的内容
    
    if not context_text:
        return "没有找到相关信息。", [], ""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "你是一位有帮助的助手。请根据给定的上下文回答问题。始终使用中文回答，无论问题是什么语言。在回答之后，请务必提供一段最相关的原文摘录，以'相关原文：'为前缀。"},
            {"role": "user", "content": f"上下文: {context_text}\n\n问题: {query}\n\n请提供你的回答然后在回答后面附上相关的原文摘录，以'相关原文：'为前缀。"}
        ]
    )
    answer = response.choices[0].message.content
    
    # 更灵活地处理回答格式
    if "相关原文：" in answer:
        answer_parts = answer.split("相关原文：", 1)
        main_answer = answer_parts[0].strip()
        relevant_excerpt = answer_parts[1].strip()
    else:
        main_answer = answer.strip()
        relevant_excerpt = ""
    
    # 如果AI没有提供相关原文，我们从上下文中选择一个
    if not relevant_excerpt and context:
        relevant_excerpt = context[0][:200] + "..."  # 使用第一个上下文的前200个字符
    
    # 找出包含相关原文的文件
    relevant_sources = []
    if relevant_excerpt:
        for file_name, chunk in context_with_sources:
            if relevant_excerpt in chunk:
                relevant_sources.append((file_name, chunk))
                break  # 只添加第一个匹配的文件
    if not relevant_sources and context_with_sources:  # 如果没有找到精确匹配，使用第一个上下文源
        relevant_sources.append(context_with_sources[0])

    return main_answer, relevant_sources, relevant_excerpt

# 保存索引和chunks
def save_index(file_name, chunks, index):
    if not os.path.exists('indices'):
        os.makedirs('indices')
    with open(f'indices/{file_name}.pkl', 'wb') as f:
        pickle.dump((chunks, index), f)
    # 更
    file_list_path = 'indices/file_list.txt'
    if os.path.exists(file_list_path):
        with open(file_list_path, 'r') as f:
            file_list = f.read().splitlines()
    else:
        file_list = []
    if file_name not in file_list:
        file_list.append(file_name)
        with open(file_list_path, 'w') as f:
            f.write('\n'.join(file_list))

# 加载所有保存的索引
def load_all_indices():
    file_indices = {}
    file_list_path = 'indices/file_list.txt'
    if os.path.exists(file_list_path):
        with open(file_list_path, 'r') as f:
            file_list = f.read().splitlines()
        for file_name in file_list:
            file_path = f'indices/{file_name}.pkl'
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    chunks, index = pickle.load(f)
                file_indices[file_name] = (chunks, index)
    return file_indices

def delete_index(file_name):
    if os.path.exists(f'indices/{file_name}.pkl'):
        os.remove(f'indices/{file_name}.pkl')
    file_list_path = 'indices/file_list.txt'
    if os.path.exists(file_list_path):
        with open(file_list_path, 'r') as f:
            file_list = f.read().splitlines()
        if file_name in file_list:
            file_list.remove(file_name)
            with open(file_list_path, 'w') as f:
                f.write('\n'.join(file_list))

def main():
    st.markdown("""
    <style>
    .reportview-container .main .block-container{
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
    .stColumn {
        padding: 0px;
    }
    .input-group {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
    }
    .button-group {
        display: flex;
        justify-content: flex-start;
        margin-top: 10px;
        width: 100%;
    }
    .button-group .stButton {
        margin-right: 10px;
    }
    .stTextInput > div > div > input {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("AI知识问答系统")

    # 初始化 session state
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []
    if "file_indices" not in st.session_state:
        st.session_state.file_indices = load_all_indices()
    if "rag_voice_input" not in st.session_state:
        st.session_state.rag_voice_input = ""
    if "web_voice_input" not in st.session_state:
        st.session_state.web_voice_input = ""
    if "db_voice_input" not in st.session_state:
        st.session_state.db_voice_input = ""

    # 创建标签
    tab1, tab2, tab3 = st.tabs(["RAG 问答", "网络搜索问答", "数据库查询"])

    with tab1:
        st.header("RAG 问答")

        # 添加CSS样式
        st.markdown("""
        <style>
        .stColumn {
            padding: 10px;
        }
        .divider {
            border-left: 2px solid #bbb;
            height: 100vh;
            position: absolute;
            left: 50%;
            margin-left: -1px;
            top: 0;
        }
        </style>
        """, unsafe_allow_html=True)

        # 创建左右两列布局
        left_column, divider, right_column = st.columns([2, 0.1, 3])

        with left_column:
            st.header("文档上传")
            
            # 设置最大token数
            max_tokens = 4096

            # 多文件上传 (添加唯一key)
            uploaded_files = st.file_uploader("上传文档", type=["pdf", "docx", "txt"], accept_multiple_files=True, key="rag_file_uploader_1")

            if uploaded_files:
                for uploaded_file in uploaded_files:
                    with st.spinner(f"正在处理文档: {uploaded_file.name}..."):
                        chunks, index = vectorize_document(uploaded_file, max_tokens)
                        st.session_state.file_indices[uploaded_file.name] = (chunks, index)
                        save_index(uploaded_file.name, chunks, index)
                    st.success(f"文档 {uploaded_file.name} 已向量化并添加到索引中！")

            # 显示已理的文档并添加删除按钮
            st.subheader("已处文档:")
            for file_name in list(st.session_state.file_indices.keys()):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"• {file_name}")
                with col2:
                    if st.button("删除", key=f"delete_{file_name}"):
                        del st.session_state.file_indices[file_name]
                        delete_index(file_name)
                        st.success(f"文档 {file_name} 已删除！")
                        st.rerun()

            # 添加关键词搜索功能
            st.subheader("关键词搜索")
            search_keywords = st.text_input("输入关键词（用空格分隔）", key="rag_search_keywords_1")
            if search_keywords:
                keywords = search_keywords.split()
                relevant_docs = search_documents(keywords, st.session_state.file_indices)
                if relevant_docs:
                    st.write("相关文档：")
                    for doc in relevant_docs:
                        st.write(f"• {doc}")
                    # 存储相关文档到 session state
                    st.session_state.relevant_docs = relevant_docs
                else:
                    st.write("没有找到相关文档。")
                    st.session_state.relevant_docs = None

        # 添加垂直分割线
        with divider:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        with right_column:
            # 创建一个容器来放置对话历史
            chat_container = st.container()

            # 显示对话历史
            with chat_container:
                for i, message in enumerate(st.session_state.rag_messages):
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                        if "sources" in message and message["sources"]:
                            st.markdown("**参考来源：**")
                            file_name, _ = message["sources"][0]
                            st.markdown(f"**文件：** {file_name}")
                            if os.path.exists(f'indices/{file_name}.pkl'):
                                with open(f'indices/{file_name}.pkl', 'rb') as f:
                                    file_content = pickle.load(f)[0]  # 获取文件内容
                                st.download_button(
                                    label="下载源文件",
                                    data='\n'.join(file_content),
                                    file_name=file_name,
                                    mime='text/plain',
                                    key=f"download_{i}"
                                )
                        if "relevant_excerpt" in message:
                            st.markdown(f"**相关原文：** <mark>{message['relevant_excerpt']}</mark>", unsafe_allow_html=True)

            # 创建一个可更新的占位符来显示语音输入状态
            status_placeholder = st.empty()

            # 创建一个按钮来触发语音输入
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            prompt = st.text_input("请基于上传的文档提出问题:", value=st.session_state.rag_voice_input, key="rag_user_input")
            st.markdown('<div class="button-group">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🎤 语音输入", key="rag_voice_input_button", help="点击开始语音输入"):
                    result = perform_speech_recognition()
                    if result:
                        st.session_state.rag_voice_input = result
                        st.rerun()
            with col2:
                if st.button("查询", key="rag_query_button"):
                    handle_rag_input()
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.header("网络搜索问答")

        # 初始化 session state
        if "web_messages" not in st.session_state:
            st.session_state.web_messages = []

        # 创建一个容器来放置对话历史
        web_chat_container = st.container()

        with web_chat_container:
            for message in st.session_state.web_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # 创建一个可更新的占位符来显示语音输入状态
        status_placeholder = st.empty()

        # 创建一个按钮来触发语音输入
        st.markdown('<div class="input-group">', unsafe_allow_html=True)
        user_input = st.text_input("输入您的问题（如需搜索，请以'搜索'开头）:", value=st.session_state.web_voice_input, key="web_user_input")
        st.markdown('<div class="button-group">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎤 语音输入", key="web_voice_input_button", help="点击开始语音输入"):
                result = perform_speech_recognition()
                if result:
                    st.session_state.web_voice_input = result
                    st.rerun()
        with col2:
            if st.button("查询", key="web_query_button"):
                handle_web_input()
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.header("自然语言数据库查询")

        # 检查数据库文件是否存在
        if not os.path.exists('chinook.db'):
            st.warning("数据库文件不存在，正在尝试下载...")
            with st.spinner("正在下载并创建数据库..."):
                download_and_create_database()
            if os.path.exists('chinook.db'):
                st.success("数据库文件已成功下载！")
            else:
                st.error("无法创建数据库文件。请检查网络连接和文件权限。")

        # 初始化 session state
        if "db_messages" not in st.session_state:
            st.session_state.db_messages = []
        if "query_result" not in st.session_state:
            st.session_state.query_result = None
        if "sql_query" not in st.session_state:
            st.session_state.sql_query = ""
        if "query_explanation" not in st.session_state:
            st.session_state.query_explanation = ""

        # 创建一个容器来放置对话历史
        db_chat_container = st.container()

        with db_chat_container:
            for message in st.session_state.db_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # 创建一个可更新的占位符来显示语音输入状态
        status_placeholder = st.empty()

        # 创建一个按钮来触发语音输入
        st.markdown('<div class="input-group">', unsafe_allow_html=True)
        nl_query = st.text_input("输入您的自然语言查询:", value=st.session_state.db_voice_input, key="db_user_input")
        st.markdown('<div class="button-group">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎤 语音输入", key="db_voice_input_button", help="点击开始语音输入"):
                result = perform_speech_recognition()
                if result:
                    st.session_state.db_voice_input = result
                    st.rerun()
        with col2:
            if st.button("查询", key="db_query_button"):
                handle_db_input()
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # 显示存储的查询结果（如果有）
        if st.session_state.sql_query:
            st.subheader("生成的SQL查询:")
            st.code(st.session_state.sql_query, language="sql")

        if st.session_state.query_result is not None:
            st.subheader("查询结果:")
            st.dataframe(st.session_state.query_result)

        if st.session_state.query_explanation:
            st.subheader("查询结果解释:")
            st.markdown(st.session_state.query_explanation, unsafe_allow_html=True)

        # 数据库信息显示部分保持不变
        if 'show_db_info' not in st.session_state:
            st.session_state.show_db_info = False
        if 'selected_table' not in st.session_state:
            st.session_state.selected_table = None

        if st.button("显示/隐藏数据库信息", key="toggle_db_info"):
            st.session_state.show_db_info = not st.session_state.show_db_info
            st.session_state.selected_table = None  # 重置选中的表

        if st.session_state.show_db_info:
            table_info = get_table_info()
            if not table_info:
                st.error("无法获取数据库信息。请检查数据库连接。")
            else:
                st.success(f"成功获取到 {len(table_info)} 个表的信息")
                
                # 创建一个动态的列布局来横向排列表名
                cols = st.columns(4)  # 每行4个表名
                for i, table in enumerate(table_info.keys()):
                    with cols[i % 4]:
                        if st.button(table, key=f"table_button_{table}"):
                            st.session_state.selected_table = table

        # 在表名下方显示查询结果
        if st.session_state.selected_table:
            st.subheader(f"表名: {st.session_state.selected_table}")
            results, column_names = execute_sql(f"SELECT * FROM {st.session_state.selected_table} LIMIT 10")
            if isinstance(results, str):
                st.error(results)
            else:
                df = pd.DataFrame(results, columns=column_names)
                st.dataframe(df)

def direct_qa(query):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "你是一个有帮助的助手，能够回答各种问题。请用中文回答。"},
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content.strip()

def serpapi_search_qa(query, num_results=3):
    params = {
        "engine": "google",
        "q": query,
        "api_key": "04fec5e75c6f477225ce29bc358f4cc7088945d0775e7f75721cd85b36387125",
        "num": num_results
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results.get("organic_results", [])
    
    if not organic_results:
        return "没有找到相关结果。"
    
    snippets = [result.get("snippet", "") for result in organic_results]
    links = [result.get("link", "") for result in organic_results]
    
    search_results = "\n".join([f"{i+1}. {snippet} ({link})" for i, (snippet, link) in enumerate(zip(snippets, links))])
    prompt = f"""问题: {query}
搜索结果:
{search_results}

请根据上述搜索结果回答问题。如果搜索结果不足以回答问题，请说"根据搜索结果无法回答问题"。"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "你是一个有帮助的助手，能够根搜索结果回答问题"},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def download_and_create_database():
    url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite"
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        with open('chinook.db', 'wb') as f:
            f.write(response.content)
        
        print("数据库文件已下载并存")
        
        conn = sqlite3.connect('chinook.db')
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = c.fetchall()
        if tables:
            print(f"成功创建数据库，包含以下表：{[table[0] for table in tables]}")
        else:
            print("数据库文件已创建，但没有找到任何表")
        conn.close()
    except Exception as e:
        print(f"下载或创建数据库时出错：{e}")

def get_table_info():
    try:
        conn = sqlite3.connect('chinook.db')
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = c.fetchall()
        
        table_info = {}
        for table in tables:
            table_name = table[0]
            c.execute(f"PRAGMA table_info({table_name})")
            columns = c.fetchall()
            table_info[table_name] = [column[1] for column in columns]
        
        conn.close()
        return table_info
    except Exception as e:
        print(f"获取表信息时出错：{e}")
        return {}

def clean_sql_query(sql_query):
    # 移除可能的 Markdown 代码块标记和多余的空白字符
    sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
    return ' '.join(sql_query.split())  # 移除余的空白字符

def nl_to_sql(nl_query):
    table_info = get_table_info()
    table_descriptions = "\n".join([f"表名: {table}\n字段: {', '.join(columns)}" for table, columns in table_info.items()])
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"你是一个SQL专，能够将自然语言查询转换为SQL语句。数据库包含以下表和字段：\n\n{table_descriptions}"},
            {"role": "user", "content": f"将以下自然语言查询转换为SQL语句：\n{nl_query}\n只返回SQL语句，不要有其他解释。"}
        ]
    )
    return clean_sql_query(response.choices[0].message.content.strip())

def execute_sql(sql_query):
    conn = sqlite3.connect('chinook.db')
    c = conn.cursor()
    try:
        sql_query = clean_sql_query(sql_query)  # 清理 SQL 查询
        c.execute(sql_query)
        results = c.fetchall()
        column_names = [description[0] for description in c.description]
        conn.close()
        return results, column_names
    except sqlite3.Error as e:
        conn.close()
        return f"SQL执行错误: {str(e)}", None

def generate_explanation(nl_query, sql_query, df):
    df_str = df.to_string(index=False, max_rows=5)
    
    prompt = (
        f"自然语言查询: {nl_query}\n"
        f"SQL查询: {sql_query}\n"
        f"查询结果 (前5行):\n"
        f"{df_str}\n\n"
        "请用通俗易懂的语言解释这个查询的结果。解释应该包括：\n"
        "1. 查询的主要目的\n"
        "2. 结果的概述\n"
        "3. 任何有趣或重要的发现\n\n"
        "请确保解释简洁明了，适合非技术人员理解。"
        "在解释中，请用**双星号**与结果直接相关的重要数字或关键词括起来。"
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "你是一个数据分析专家，擅长解释SQL查询结果。"},
            {"role": "user", "content": prompt}
        ]
    )
    explanation = response.choices[0].message.content.strip()
    
    # 将双星号包围的文本转换为加粗文本
    explanation = explanation.replace("**", "**")
    
    return explanation

def perform_speech_recognition():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.write("正在录音...请说话")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
        st.write("录音完成，正在识别...")
        try:
            text = recognizer.recognize_google(audio, language="zh-CN")
            return text
        except sr.UnknownValueError:
            st.error("无法识别语音，请重试")
        except sr.RequestError as e:
            st.error(f"无法从Google Speech Recognition服务获取结果; {e}")
    except Exception as e:
        st.error(f"录音过程中出现错误: {str(e)}")
    return None

def handle_rag_input():
    if st.session_state.rag_user_input:
        prompt = st.session_state.rag_user_input
        st.session_state.rag_messages.append({"role": "user", "content": prompt})
        if st.session_state.file_indices:
            with st.spinner("正在生成回答..."):
                try:
                    relevant_docs = st.session_state.get('relevant_docs')
                    response, sources, relevant_excerpt = rag_qa(prompt, st.session_state.file_indices, relevant_docs)
                    st.session_state.rag_messages.append({
                        "role": "assistant", 
                        "content": response, 
                        "sources": sources,
                        "relevant_excerpt": relevant_excerpt
                    })
                except Exception as e:
                    st.error(f"生成回答时发生错误: {str(e)}")
        else:
            st.warning("请先上传文档。")
        st.session_state.rag_voice_input = ""
        st.rerun()

def handle_web_input():
    if st.session_state.web_user_input:
        user_input = st.session_state.web_user_input
        st.session_state.web_messages.append({"role": "user", "content": user_input})
        with st.spinner("正在搜索并生成回答..."):
            try:
                if user_input.lower().startswith("搜索"):
                    response = serpapi_search_qa(user_input[2:].strip())
                else:
                    response = direct_qa(user_input)
                st.session_state.web_messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"生成回答时发生错误: {str(e)}")
        st.session_state.web_voice_input = ""
        st.rerun()

def handle_db_input():
    if st.session_state.db_user_input:
        nl_query = st.session_state.db_user_input
        st.session_state.db_messages.append({"role": "user", "content": nl_query})
        with st.spinner("正在生成SQL并执行查询..."):
            try:
                sql_query = nl_to_sql(nl_query)
                st.session_state.sql_query = sql_query
                results, column_names = execute_sql(sql_query)
                if isinstance(results, str):
                    st.error(results)
                else:
                    df = pd.DataFrame(results, columns=column_names)
                    st.session_state.query_result = df
                    explanation = generate_explanation(nl_query, sql_query, df)
                    st.session_state.query_explanation = explanation
                    response = f"SQL查询:\n```sql\n{sql_query}\n```\n\n查询结果:\n{df.to_markdown()}\n\n解释:\n{explanation}"
                    st.session_state.db_messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"查询执行错误: {str(e)}")
        st.session_state.db_voice_input = ""
        st.rerun()

# 运行主应用
if __name__ == "__main__":
    main()
