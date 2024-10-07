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
   streamlit
   openai
   sentence-transformers
   PyPDF2
   python-docx
   faiss-cpu
   tiktoken
   serpapi
   pandas
   sqlite3  # 通常已包含在Python标准库中

5. 其他依赖：
   - 确保你有有效的OpenAI API密钥
   - 如果使用Google搜索功能，需要有效的SerpAPI密钥

6. 运行应用：
   streamlit run app.py

注意：
- 请确保所有依赖都已正确安装
- 在代码中替换OpenAI API密钥和SerpAPI密钥为你自己的密钥
- 对于大型文件处理，可能需要增加系统内存或使用更强大的硬件
"""

import streamlit as st
import sys
import os
import logging
import json

# 设置页面配置必须是第一个 Streamlit 命令
st.set_page_config(
    page_title="AI知识问答系统 - by Huaiyuan Tan",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None
)

# 添加开发者信息
st.markdown("<h6 style='text-align: right; color: gray;'>开发者: Huaiyuan Tan</h6>", unsafe_allow_html=True)

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
import pandas as pd
from neo4j_operations import get_neo4j_connection, AURA_CONFIG, LOCAL_CONFIG
from data_processor import export_to_neo4j, query_imported_data, DataProcessor
from graph_builder import GraphBuilder
from graph_query import GraphQuery

# 初始化
client = OpenAI(
    api_key="sk-1pUmQlsIkgla3CuvKTgCrzDZ3r0pBxO608YJvIHCN18lvOrn",
    base_url="https://api.chatanywhere.tech/v1"
)

@st.cache_resource
def load_model():
    with st.spinner('正在加载语言模型，这可能需要几分钟...'):
        return SentenceTransformer('paraphrase-MiniLM-L3-v2')  # 使用更小的模型

# 全局变量来存储模型
model = None

# 计算token数量
def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(string))

# 文档向量化模
def vectorize_document(file, max_tokens, model):
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
    # 过滤掉停用词和个字符
    keywords = [word for word, count in word_count.most_common(top_k*2) if len(word) > 1]
    return keywords[:top_k]

# 新增函数：基于关键词搜索文档
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
        if 0 <= i < len(all_chunks):  # 确索引在有效范围内
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
            {"role": "system", "content": "你是一位有帮助的助手。请根据给定的上下文回答问题。始终使用中文回答，无论问题是什么语言。在回答之后，请务必提供一个最相关的原文摘录，以'相关原文：'为前缀。"},
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
    # 保存文件名到一个列表中
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

def combined_qa(query, file_indices, neo4j_conn, relevant_docs=None):
    # 使用FAISS检索
    faiss_response, faiss_sources, faiss_excerpt = rag_qa(query, file_indices, relevant_docs)
    
    # 使用Neo4j检索
    graph_query = GraphQuery(neo4j_conn)
    neo4j_response, neo4j_sources, neo4j_excerpt = graph_query.neo4j_qa(query)
    
    # 合并结果
    combined_context = f"FAISS结果：{faiss_response}\n\nNeo4j结果：{neo4j_response}"
    
    # 使用合并后的上下文生成最终答案
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "你是一位有帮助的助手。请根据给定的FAISS和Neo4j的结果，综合分析并给出最终答案。始终使用中文回答，无论问题是什么语言。在回答之后，请务必提供一个最相关的原文摘录，以'相关原文：'为前缀。"},
            {"role": "user", "content": f"上下文: {combined_context}\n\n问题: {query}\n\n请提供你的综合回答，然后在回答后面附上最相关的原文摘录，以'相关原文：'为前缀。"}
        ]
    )
    final_answer = response.choices[0].message.content
    
    # 处理回答格式
    if "相关原文：" in final_answer:
        answer_parts = final_answer.split("相关原文：", 1)
        main_answer = answer_parts[0].strip()
        relevant_excerpt = answer_parts[1].strip()
    else:
        main_answer = final_answer.strip()
        relevant_excerpt = faiss_excerpt or neo4j_excerpt
    
    # 合并源
    combined_sources = list(set(faiss_sources + neo4j_sources))
    
    return main_answer, combined_sources, relevant_excerpt

def main():
    global model  # 声明使用全局变量
    
    st.markdown("""
    <style>
    .reportview-container .main .block-container{
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
    .stColumn {
        padding: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("AI知识问答系统")

    # 加Neo4j连接选择
    st.sidebar.title("Neo4j连接设置")
    connection_type = st.sidebar.radio("择Neo4j连接类型", ("本地", "Aura"))
    
    if connection_type == "本地":
        neo4j_config = LOCAL_CONFIG
    else:
        neo4j_config = AURA_CONFIG
    
    # 创建Neo4j连接
    try:
        neo4j_conn = get_neo4j_connection(neo4j_config)
        
        # 执行简单查询来验证连接
        result = neo4j_conn.query("CALL dbms.components() YIELD name, versions, edition UNWIND versions as version RETURN name, version, edition;")
        if result:
            db_info = result[0]
            st.sidebar.success(f"成功连接到 {connection_type} Neo4j数据库")
            st.sidebar.info(f"数据库名称: {db_info['name']}")
            st.sidebar.info(f"版本: {db_info['version']}")
            st.sidebar.info(f"版本: {db_info['edition']}")
        else:
            st.sidebar.warning(f"已连接到 {connection_type} Neo4j数据库，但无法获取数据库信息")

        # 添加测试按钮
        if st.sidebar.button("测试Neo4j连接"):
            try:
                # 创建测试节点
                create_result = neo4j_conn.create_node("TestNode", {"name": "Test", "timestamp": str(pd.Timestamp.now())})
                st.sidebar.success("成功创建测试节点")

                # 查询所有TestNode
                query_result = neo4j_conn.query("MATCH (n:TestNode) RETURN n")
                st.sidebar.write("查询结果:")
                for record in query_result:
                    st.sidebar.write(record['n'])

            except Exception as e:
                st.sidebar.error(f"测试Neo4j连接时出错: {str(e)}")

    except Exception as e:
        st.sidebar.error(f"连接 {connection_type} Neo4j数据库失败: {str(e)}")
        neo4j_conn = None

    # 初始化 session state
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []
    if "file_indices" not in st.session_state:
        st.session_state.file_indices = load_all_indices()
    
    # 提前加载模型
    model = load_model()

    st.header("RAG 问答")

    # 文档上传部分
    st.subheader("文档上传")
    
    max_tokens = 4096

    uploaded_files = st.file_uploader("上传文档", type=["pdf", "docx", "txt"], accept_multiple_files=True, key="rag_file_uploader_1")

    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.spinner(f"正在处理文档: {uploaded_file.name}..."):
                chunks, index = vectorize_document(uploaded_file, max_tokens, model)  # 传入 model
                st.session_state.file_indices[uploaded_file.name] = (chunks, index)
                save_index(uploaded_file.name, chunks, index)
            st.success(f"文档 {uploaded_file.name} 已向量化并添加到索引中！")

    # 显示已处理的文件并添加删除按钮
    st.subheader("已处理文档:")
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
            st.session_state.relevant_docs = relevant_docs
        else:
            st.write("没有找到相关文档。")
            st.session_state.relevant_docs = None

    # 添加检索方法选择
    st.sidebar.subheader("检索方法选择")
    retrieval_method = st.sidebar.radio(
        "选择检索方法",
        ("FAISS", "Neo4j", "FAISS + Neo4j")
    )

    # 对话部分
    st.subheader("对话")
    chat_container = st.container()

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
                            label="下载源文",
                            data='\n'.join(file_content),
                            file_name=file_name,
                            mime='text/plain',
                            key=f"download_{i}"
                        )
                if "relevant_excerpt" in message:
                    st.markdown(f"**相关原文：** <mark>{message['relevant_excerpt']}</mark>", unsafe_allow_html=True)

    # 用户输入
    prompt = st.chat_input("请基于上传的文档提出题:", key="rag_chat_input_1")

    if prompt:
        st.session_state.rag_messages.append({"role": "user", "content": prompt})
        
        if st.session_state.file_indices:
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    with st.spinner("正在生成回答..."):
                        try:
                            relevant_docs = st.session_state.get('relevant_docs')
                            
                            # 根据选择的检索方法执行不同的检索
                            if retrieval_method == "FAISS":
                                response, sources, relevant_excerpt = rag_qa(prompt, st.session_state.file_indices, relevant_docs)
                            elif retrieval_method == "Neo4j":
                                graph_query = GraphQuery(neo4j_conn)
                                response, sources, relevant_excerpt = graph_query.neo4j_qa(prompt)
                            else:  # FAISS + Neo4j
                                response, sources, relevant_excerpt = combined_qa(prompt, st.session_state.file_indices, neo4j_conn, relevant_docs)
                            
                            st.markdown(response)
                            if sources:
                                st.markdown("**参考来源：**")
                                file_name, _ = sources[0]
                                st.markdown(f"**文件：** {file_name}")
                                if os.path.exists(f'indices/{file_name}.pkl'):
                                    with open(f'indices/{file_name}.pkl', 'rb') as f:
                                        file_content = pickle.load(f)[0]  # 获取文件内容
                                    st.download_button(
                                        label="下载源文件",
                                        data='\n'.join(file_content),
                                        file_name=file_name,
                                        mime='text/plain',
                                        key=f"download_new_{len(st.session_state.rag_messages)}"
                                    )
                            if relevant_excerpt:
                                st.markdown(f"**相关原文：** <mark>{relevant_excerpt}</mark>", unsafe_allow_html=True)
                            else:
                                st.warning("未能提取到精确的相关原文，但找到相关信息。")
                        except Exception as e:
                            st.error(f"生成回答时发生错误: {str(e)}")
                st.session_state.rag_messages.append({
                    "role": "assistant", 
                    "content": response, 
                    "sources": sources,
                    "relevant_excerpt": relevant_excerpt
                })
        else:
            with chat_container:
                with st.chat_message("assistant"):
                    st.warning("请先上传文档。")

    if st.sidebar.button("导出数据到Neo4j并构建图谱"):
        if neo4j_conn and st.session_state.file_indices:
            try:
                # 设置日志级别
                logging.basicConfig(level=logging.INFO)
                logger = logging.getLogger(__name__)

                # 使用 DataProcessor 处理数据
                processor = DataProcessor()
                processed_data = processor.process_file_indices(st.session_state.file_indices)
                
                # 添加日志来检查处理后的数据
                logger.info(f"Processed data: {processed_data}")

                # 使用 GraphBuilder 构建图谱
                graph_builder = GraphBuilder(neo4j_conn)
                node_count, rel_count, next_relationships_count = graph_builder.build_graph(processed_data)

                st.sidebar.success(f"成功导出数据到Neo4j并构建图谱。")
                st.sidebar.info(f"创建的TextChunk节点数量: {node_count}")
                st.sidebar.info(f"创建的NEXT关系数量: {rel_count}")
                st.sidebar.info(f"尝试创建的NEXT关系数量: {next_relationships_count}")

                # 显示一些示例节点和关系
                st.sidebar.subheader("示例节点:")
                sample_nodes = neo4j_conn.query("MATCH (n:TextChunk) RETURN n LIMIT 5")
                for node in sample_nodes:
                    try:
                        node_data = dict(node['n'])
                        # 如果 embedding 是字符串形式的 JSON，我们可以尝试解析它
                        if 'embedding' in node_data and isinstance(node_data['embedding'], str):
                            node_data['embedding'] = json.loads(node_data['embedding'])
                        st.sidebar.json(node_data)
                    except Exception as e:
                        st.sidebar.error(f"Error displaying node: {str(e)}")

                st.sidebar.subheader("示例关系:")
                sample_relationships = neo4j_conn.query("MATCH (a:TextChunk)-[r:NEXT]->(b:TextChunk) RETURN a.chunk_id, b.chunk_id, type(r) LIMIT 5")
                for rel in sample_relationships:
                    st.sidebar.write(f"{rel['a.chunk_id']} -[{rel['type(r)']}]-> {rel['b.chunk_id']}")

            except Exception as e:
                st.sidebar.error(f"导出数据到Neo4j并构建图谱时出错: {str(e)}")
                logger.error(f"Error during graph building: {str(e)}", exc_info=True)
        else:
            st.sidebar.warning("请确保已连接到Neo4j并上传了文档。")

    # 在main函数结束前关闭Neo4j连接
    if neo4j_conn:
        neo4j_conn.close()

# 在文件末尾添加
if __name__ == "__main__":
    main()
else:
    # 导出关键函数
    __all__ = ['load_model', 'vectorize_document', 'save_index', 'load_all_indices', 'delete_index', 'rag_qa']