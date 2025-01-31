"""
基于图数据库及向量数据库的混合问答系统
"""

import streamlit as st
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
import torch
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
import io
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import re  # 添加这一行
from data_processor import (
    load_model, vectorize_document, extract_keywords, 
    search_documents, save_index, load_all_indices, 
    delete_index, rag_qa, initialize_openai,
    query_graph, hybrid_search, get_entity_relations,
    set_neo4j_config, get_neo4j_driver, process_data,
    generate_final_answer, vector_search, execute_neo4j_query,
    initialize_faiss, create_fulltext_index, search_fulltext_index,
    open_dir, delete_graph_data, delete_vector_data, delete_fulltext_index,
    clear_vector_data, Term, initialize_neo4j, CURRENT_NEO4J_CONFIG
)
from whoosh.query import Term

# 在文件顶部的导入语句之后添加
from data_processor import faiss_id_to_text, faiss_id_counter, faiss_index

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key="EMPTY",  # Ollama 不需要 API key
    base_url="http://152.70.248.22:1234/api/chat"  # Ollama API 地址
)

# 设置面配置
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

# 初始化 OpenAI 客户端
initialize_openai(
    api_key="EMPTY",  # Ollama 不需要 API key
    base_url="http://152.70.248.22:1234"
)

# 初始化 session state
if "file_indices" not in st.session_state:
    st.session_state.file_indices = load_all_indices()

def decompose_query(query):
    prompt = f"""
    请将以下复杂查询分解为多个简单子查询：
    {query}
    
    输出格式：
    1. 子查询1
    2. 子查询2
    ...
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "你是一专门于分解复杂查询的AI助手。"},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip().split("\n")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class CustomSentenceTransformer:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
    
    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            encoded_input = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            all_embeddings.extend(sentence_embeddings)
        return torch.stack(all_embeddings).numpy()

# 使用自定义的 SentenceTransformer
SentenceTransformer = CustomSentenceTransformer

def main():
    # 设置 Neo4j 配置
    try:
        # 在使用 neo4j_option 之前先定义它
        col1, col2, col3 = st.columns(3)
        with col1:
            neo4j_option = st.radio(
                "选择 Neo4j 连接方式",
                ("Neo4j Aura", "本地 Neo4j")
            )

        # 根据选择设置配置
        if neo4j_option == "Neo4j Aura":
            config = set_neo4j_config("AURA")
        else:
            config = set_neo4j_config("LOCAL")
        
        # 初始化 Neo4j
        initialize_neo4j()

        with col2:
            if neo4j_option == "Neo4j Aura":
                connection_status = "已选择连接到 Neo4j Aura"
            else:
                connection_status = "已选择连接到本地 Neo4j"

        with col3:
            # 测试数据库连接
            try:
                driver = get_neo4j_driver()
                with driver.session() as session:
                    result = session.run("RETURN 1 AS test")
                    test_value = result.single()["test"]
                    if test_value == 1:
                        connection_status += " - 连接成功"
                    else:
                        connection_status += " - 连接测试失败"
                driver.close()
            except Exception as e:
                connection_status += f" - 连接错误: {str(e)}"
                st.error(f"数据库连接错误: {str(e)}")

        st.write(connection_status)

    except Exception as e:
        st.error(f"初始化 Neo4j 时出错: {str(e)}")
        return

    # 添加标题和开发者信息
    st.markdown(
        """
        <style>
        .title-container {
            display: flex;
            align-items: baseline;
        }
        .main-title {
            font-size: 2em;
            font-weight: bold;
        }
        .developer-info {
            font-size: 1em;
            color: #888;
            font-weight: bold;
            margin-left: 100px;
        }
        </style>
        <div class="title-container">
            <span class="main-title">AI知识库问答系统</span>
            <span class="developer-info">Developed by Huaiyuan Tan</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    if 'faiss_index' not in st.session_state:
        st.session_state.faiss_index = initialize_faiss()
    
    if 'faiss_id_to_text' not in st.session_state:
        st.session_state.faiss_id_to_text = {}
    if 'faiss_id_counter' not in st.session_state:
        st.session_state.faiss_id_counter = 0
    
    # 初始化 FAISS
    try:
        faiss_index = initialize_faiss()
        if faiss_index is None:
            st.error("FAISS 索引初始化失败。请查 initialize_faiss() 数。")
            return
    except Exception as e:
        st.error(f"FAISS 索引初始化时发生错误: {str(e)}")
        return

    # 加载所有索引
    st.session_state.file_indices = load_all_indices()
    
    # 如果有索引，将它们添加到 FAISS 索引中
    if st.session_state.file_indices:
        for file_name, (chunks, index, patient_name) in st.session_state.file_indices.items():
            for i, chunk in enumerate(chunks):
                st.session_state.faiss_id_to_text[st.session_state.faiss_id_counter + i] = chunk
            vectors = index.reconstruct_n(0, index.ntotal)
            faiss_index.add(vectors)
        st.session_state.faiss_id_counter += sum(len(chunks) for chunks, _, _ in st.session_state.file_indices.values())

    # 创建四个标签页
    tab1, tab2, tab3, tab4 = st.tabs(["文档上传", "知识库问答", "数据库检索", "数据管理"])

    with tab1:
        st.header("文档上传")
        
        # 设置最大token数
        max_tokens = 4096

        # 多文件上传
        uploaded_files = st.file_uploader("传文档", type=["pdf", "docx", "txt"], accept_multiple_files=True)

        if uploaded_files:
            for uploaded_file in uploaded_files:
                content = uploaded_file.read()
                if uploaded_file.type == "application/pdf":
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
                    content = ""
                    for page in pdf_reader.pages:
                        content += page.extract_text()
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    doc = docx.Document(io.BytesIO(content))
                    content = "\n".join([para.text for para in doc.paragraphs])
                else:
                    content = content.decode('utf-8')
                
                st.write(f"文件 '{uploaded_file.name}' 已上传")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("加载到图数据库", key=f"graph_{uploaded_file.name}"):
                        with st.spinner(f"正在处理文档并加载到图数据库: {uploaded_file.name}..."):
                            # 先删除旧数据
                            delete_graph_data(uploaded_file.name)
                            # 添加新数据
                            result = process_data(content, uploaded_file.name)
                        st.success(f"文 {uploaded_file.name} 已成功加载到图数据库！")
                        st.write(f"处理了 {len(result['entities'])} 个实体和 {len(result['relations'])} 个关系")
                        
                        # 示处理结果的详细信息
                        with st.expander("查看详细处理结果"):
                            st.subheader("实体:")
                            for entity in result['entities']:
                                st.write(f"- {entity}")
                            st.subheader("关系:")
                            for relation in result['relations']:
                                st.write(f"- {relation['source']} --[{relation['relation']}]--> {relation['target']}")
                        
                        # 生成并显示关系图
                        G = nx.Graph()
                        for entity in result['entities']:
                            G.add_node(entity['name'], category=entity['category'])
                        for relation in result['relations']:
                            G.add_edge(relation['source'], relation['target'], relation=relation['relation'])
                        
                        net = Network(notebook=True, width="100%", height="500px", bgcolor="#222222", font_color="white")
                        net.from_nx(G)
                        net.toggle_physics(True)
                        net.show_buttons(filter_=['physics'])
                        net.save_graph("temp_graph.html")
                        
                        with open("temp_graph.html", "r", encoding="utf-8") as f:
                            graph_html = f.read()
                        
                        st.subheader("电子病历关系图")
                        components.html(graph_html, height=600)
                
                with col2:
                    if st.button("加载到向量数据库", key=f"vector_{uploaded_file.name}"):
                        with st.spinner(f"正在处理文档并加载到向量数据库: {uploaded_file.name}..."):
                            # 先删除旧数据
                            if uploaded_file.name in st.session_state.file_indices:
                                delete_vector_data(uploaded_file.name)
                            # 添加新数据
                            chunks, index, patient_name = vectorize_document(content, uploaded_file.name, max_tokens)
                            st.session_state.file_indices[uploaded_file.name] = (chunks, index, patient_name)
                            save_index(uploaded_file.name, chunks, index, patient_name)
                        st.success(f"文档 {uploaded_file.name} 已成功加载到向量数据库！")
                        st.write(f"向 FAISS 向量数据库添加了 {len(chunks)} 个文本段落")
                        st.write(f"患者姓名: {patient_name}")
                
                with col3:
                    if st.button("创建全文索引", key=f"fulltext_{uploaded_file.name}"):
                        with st.spinner(f"正在为文档创建全文索引: {uploaded_file.name}..."):
                            try:
                                # 先删除旧索引
                                delete_fulltext_index(uploaded_file.name)
                                # 创索引
                                create_fulltext_index(content, uploaded_file.name)
                                st.success(f"文档 {uploaded_file.name} 已成功创建全文索引！")
                                # 添加验证步骤
                                ix = open_dir("fulltext_index")
                                with ix.searcher() as searcher:
                                    results = searcher.search(Term("title", uploaded_file.name))
                                    if results:
                                        st.success(f"成功验证文档 {uploaded_file.name} 已被索引")
                                        st.info(f"索引中的文档内容长度: {len(results[0]['content'])}")
                                        st.info(f"索引中的文档内前200字符: {results[0]['content'][:200]}")
                                    else:
                                        st.warning(f"无法在索引找到文档 {uploaded_file.name}")
                            except Exception as e:
                                st.error(f"创建或验证索时出错: {str(e)}")
                                st.error(f"错误类型: {type(e).__name__}")
                                st.error(f"错误详情: {e.args}")

        # 显示已处理的文档并添加删除按钮
        st.subheader("已处理文件:")
        for file_name in list(st.session_state.file_indices.keys()):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"• {file_name}")
            with col2:
                if st.button("删除", key=f"delete_{file_name}"):
                    with st.spinner(f"正在删除文档 {file_name} 的所有相关数据..."):
                        try:
                            # 删除向量数据库中的数据
                            delete_vector_data(file_name)
                            
                            # 删除图数据库中的数据
                            delete_graph_data(file_name)
                            
                            # 删除全索引中的数据
                            delete_fulltext_index(file_name)
                            
                            # 删除本地索引文件
                            delete_index(file_name)
                            
                            # 验证删除操作
                            ix = open_dir("fulltext_index")
                            with ix.searcher() as searcher:
                                remaining_docs = [doc for doc in searcher.all_stored_fields() if doc['title'].startswith(file_name)]
                                if not remaining_docs:
                                    st.success(f"文档 {file_name} 及其所有相关数据已成功删除！")
                                else:
                                    st.warning(f"文档 {file_name} 的一些相关数据可能没有完全删除。正在尝试清理...")
                                    for doc in remaining_docs:
                                        delete_fulltext_index(doc['title'])
                                    st.success(f"清理完成。请检查索引以确保所有相关数据已被删除。")
                        except Exception as e:
                            st.error(f"删除文档 {file_name} 时出错: {str(e)}")
                            logger.error(f"删除文档 {file_name} 时出错", exc_info=True)
                    st.rerun()

    with tab2:
        st.header("知识库问答")
        
        qa_type = st.radio("选择问答类型", ["向量数据库问答", "图数据库问答", "全文索引问答", "混合问答"])
        
        if qa_type == "向量数据库问答":
            st.subheader("向量数据库问答")
            with st.form(key='vector_qa_form'):
                vector_query = st.text_input("请输入您的问题（向量数据库）")
                submit_button = st.form_submit_button(label='提交问题')
            if submit_button and vector_query:
                with st.spinner("正在查询..."):
                    answer, sources, excerpt = rag_qa(vector_query, st.session_state.file_indices,k=10)
                st.write("回答：", answer)
                if sources:
                    st.write("参考来源：")
                    for source in sources:
                        st.write(f"- 文件: {source['file_name']}, 患者: {source['patient_name']}")
                if excerpt:
                    st.write("相原文：")
                    st.write(excerpt)
        
        elif qa_type == "图数据库问答":
            st.subheader("图数据库问答")
            with st.form(key='graph_qa_form'):
                graph_query = st.text_input("请输入您的问题（图数据库）")
                submit_button = st.form_submit_button(label='提交问题')
            if submit_button and graph_query:
                with st.spinner("正在查询..."):
                    answer, entities, relations = hybrid_search(graph_query)
                    
                    # 使用 OpenAI API 生成综合回答
                    context = f"""
                    基于图数据库的查询结果：
                    回答：{answer}
                    实体：{', '.join(entities)}
                    关系：{'; '.join([f"{r['source']} --[{r['relation']}]--> {r['target']}" for r in relations])}
                    
                    请根据以上信息，生成一个简洁明了的综合回答。回答应该直接针对问题"{graph_query}"，并包含所有相关的重要信息。
                    """
                    
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "你是一个专门解释图数据库查询结果的AI助手。请提供准确、简洁的回答。"},
                            {"role": "user", "content": context}
                        ],
                        max_tokens=200
                    )
                    
                    final_answer = response.choices[0].message.content.strip()
                    
                    st.write("综合回答：", final_answer)
                    
                    st.write("图数据库查询结果：")
                    st.write("实体：", ", ".join(entities))
                    st.write("关系：")
                    for relation in relations:
                        st.write(f"- {relation['source']} --[{relation['relation']}]--> {relation['target']}")
                    
                    # 创建并显示关系图
                    G = nx.Graph()
                    for entity in entities:
                        G.add_node(entity)
                    for relation in relations:
                        G.add_edge(relation['source'], relation['target'], title=relation['relation'])
                    
                    net = Network(notebook=True, width="100%", height="500px", bgcolor="#222222", font_color="white")
                    net.from_nx(G)
                    net.save_graph("graph.html")
                    
                    with open("graph.html", 'r', encoding='utf-8') as f:
                        html_string = f.read()
                    st.write("图数据库关系图谱：")
                    components.html(html_string, height=600)
        
        elif qa_type == "全文索引问答":
            st.subheader("全文索引问答")
            with st.form(key='fulltext_qa_form'):
                fulltext_query = st.text_input("请输入您的问题（全文索引）")
                submit_button = st.form_submit_button(label='提交问题')
            if submit_button and fulltext_query:
                with st.spinner("正在查询..."):
                    try:
                        fulltext_results = search_fulltext_index(fulltext_query)
                        if fulltext_results:
                            # 准备上下文
                            context = "\n\n".join([f"文档: {result['title']}\n内容: {result['content']}" for result in fulltext_results[:3]])
                            
                            # 使用 OpenAI API 生成总结答案
                            prompt = f"""基于以下从全文索引中检索到的信息，回答问题并提供简明了的总结：

问题：{fulltext_query}

检索到的信息：
{context}

请提供一个综合的回答，包括：
1. 直接回答问题
2. 对检索到的信息进行简总结
3. 如果信息不足以完全回答问题，请说明并提供可能的下一步建议

回答："""

                            response = client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "system", "content": "你是一个专门用于总结和回答问题的AI助手。请基于给定的信息提供准确、简洁的回答。"},
                                    {"role": "user", "content": prompt}
                                ],
                                max_tokens=300
                            )

                            summary = response.choices[0].message.content.strip()
                            
                            # 显示总结答案
                            st.write("回答：")
                            st.write(summary)
                            
                            # 显示原始检索结果
                            st.write("\n原始检索结果：")
                            for result in fulltext_results[:3]:  # 显示前3个结果
                                st.write(f"- 文档: {result['title']}, 相关度: {result['score']:.2f}")
                                highlights = result['highlights']
                                # 处理高亮文本
                                highlights = re.sub(r'<b class="match term\d+">', '**', highlights)
                                highlights = highlights.replace('</b>', '**')
                                # 将连续的星号合并
                                highlights = re.sub(r'\*{2,}', '**', highlights)
                                # 移除可能残留的HTML标签
                                highlights = re.sub(r'<[^>]+>', '', highlights)
                                st.markdown(f"  匹配内容: {highlights}")
                                st.write(f"  文档内容片段: {result['content'][:200]}...")  # 只显示前200个字符
                        else:
                            st.write("全文检索未找到相关结果。")
                    except Exception as e:
                        st.error(f"全文检索出错: {str(e)}")

        else:  # 混合问答
            st.subheader("混合问答")
            with st.form(key='hybrid_qa_form'):
                hybrid_query = st.text_input("请输入您的问题（混合问答）")
                submit_button = st.form_submit_button(label='提交问题')
            if submit_button and hybrid_query:
                with st.spinner("正在查询..."):
                    # 全文检索
                    try:
                        fulltext_results = search_fulltext_index(hybrid_query)
                    except Exception as e:
                        st.error(f"全文检索出错: {str(e)}")
                        fulltext_results = []

                    # 图数据库查询
                    graph_answer, graph_entities, graph_relations = hybrid_search(hybrid_query)
                    
                    # 向量数据库查询
                    vector_answer, sources, excerpt = rag_qa(hybrid_query, st.session_state.file_indices)
                    
                    # 使用所有结果生成最终答案
                    final_answer = generate_final_answer(
                        hybrid_query, 
                        graph_answer, 
                        vector_answer, 
                        fulltext_results,  # 确保这里传递的是完整的 fulltext_results
                        excerpt, 
                        graph_entities, 
                        graph_relations
                    )
                    
                    st.write("最终回答：", final_answer)
                    
                    # 图数据库回答
                    st.write("图数据库回答：", graph_answer)
                    
                    # 创并显示关系图谱
                    G = nx.Graph()
                    for entity in graph_entities:
                        G.add_node(entity)
                    for relation in graph_relations:
                        G.add_edge(relation['source'], relation['target'], title=relation['relation'])
                    
                    net = Network(notebook=True, width="100%", height="500px", bgcolor="#222222", font_color="white")
                    net.from_nx(G)
                    net.toggle_physics(True)
                    net.show_buttons(filter_=['physics'])
                    net.save_graph("graph.html")
                    
                    with open("graph.html", 'r', encoding='utf-8') as f:
                        html_string = f.read()
                    st.write("图数据库关系图谱：")
                    components.html(html_string, height=600)
                    
                    # 向量数据库回答
                    st.write("向量数据库回答：", vector_answer)
                    
                    # 显示全文检索结果
                    if fulltext_results:
                        st.write("全文检索结果（前3个）：")
                        for result in fulltext_results[:3]:  # 显示前3个结果
                            st.write(f"- 文档: {result['title']}, 相关度: {result['score']:.2f}")
                            highlights = result['highlights']
                            # 处理高亮文本
                            highlights = re.sub(r'<b class="match term\d+">', '**', highlights)
                            highlights = highlights.replace('</b>', '**')
                            # 将连续的星号合并
                            highlights = re.sub(r'\*{2,}', '**', highlights)
                            # 移除可能残留的HTML标签
                            highlights = re.sub(r'<[^>]+>', '', highlights)
                            st.markdown(f"  匹配内容: {highlights}")
                            st.write(f"  文档内容片段: {result['content'][:200]}...")  # 只显示前200个字符
                    else:
                        st.write("全文检索未找到相关结果。")

        # 添加关键词搜索功能
        st.subheader("关键词搜索")
        with st.form(key='keyword_search_form'):
            search_keywords = st.text_input("输入关键词（用空格分隔）")
            submit_button = st.form_submit_button(label='搜索')
        if submit_button and search_keywords:
            keywords = search_keywords.split()
            relevant_docs = search_documents(keywords, st.session_state.file_indices)
            if relevant_docs:
                st.write("相关文档：")
                for doc in relevant_docs:
                    st.write(f" {doc}")
                # 存储相关文档到 session state
                st.session_state.relevant_docs = relevant_docs
            else:
                st.write("没有找到相关文档。")
                st.session_state.relevant_docs = None

    with tab3:
        st.header("数据库检索")
        
        search_type = st.radio("选择搜索类", ["图数据库索", "向量数据库搜索", "全文索引搜索", "Neo4j 命令执行"])
        
        if search_type == "图数据库索":
            st.subheader("图数库搜索")
            with st.form(key='graph_search_form'):
                graph_query = st.text_input("输入搜索关键词")
                submit_button = st.form_submit_button(label='执行图数据库搜索')
            if submit_button and graph_query:
                with st.spinner("正在搜索图数据库..."):
                    entities, relations, contents = query_graph(graph_query)
                if entities or relations:
                    st.success("搜索完成")
                    st.write("找到的实体:")
                    st.write(", ".join(entities))
                    st.write("相关关系:")
                    for relation in relations:
                        st.write(f"{relation['source']} --[{relation['relation']}]--> {relation['target']}")
                    
                    # 创建并显示关系图
                    G = nx.Graph()
                    for entity in entities:
                        G.add_node(entity)
                    for relation in relations:
                        G.add_edge(relation['source'], relation['target'], title=relation['relation'])
                    
                    net = Network(notebook=True, width="100%", height="500px", bgcolor="#222222", font_color="white")
                    net.from_nx(G)
                    net.toggle_physics(True)
                    net.show_buttons(filter_=['physics'])
                    net.save_graph("graph.html")
                    
                    with open("graph.html", 'r', encoding='utf-8') as f:
                        html_string = f.read()
                    components.html(html_string, height=600)
                else:
                    st.warning("没有找到相关信息。")

        elif search_type == "向量数据库搜索":
            st.subheader("向量数据库搜索")
            with st.form(key='vector_search_form'):
                vector_query = st.text_input("输入搜索关键词")
                submit_button = st.form_submit_button(label='执行向量数据库搜索')
            if submit_button and vector_query:
                with st.spinner("正在搜索向量数据库..."):
                    results = vector_search(vector_query, k=5)  # 假设 k=5，返前5个最相似的结果
                if results:
                    st.success("搜索完成！")
                    for i, result in enumerate(results, 1):
                        st.write(f"结果 {i}:")
                        st.write(f"相似度: {1 - result['distance']:.4f}")
                        st.write(f"内容: {result['text'][:200]}...")  # 只显示前200个字符
                        st.write("---")
                else:
                    st.warning("没有找到相关信息。")

        elif search_type == "全文索引搜索":
            st.subheader("全文索引搜索")
            with st.form(key='fulltext_search_form'):
                fulltext_query = st.text_input("输搜索关键词")
                submit_button = st.form_submit_button(label='执行全文索引搜索')
            if submit_button and fulltext_query:
                with st.spinner("正在搜索全文索引..."):
                    try:
                        results = search_fulltext_index(fulltext_query)
                        if results:
                            st.success(f"找到 {len(results)} 个相关文档")
                            for result in results:
                                st.write(f"文档: {result['title']}, 关度: {result['score']:.2f}")
                                highlights = result['highlights']
                                # 处理高亮文本
                                highlights = re.sub(r'<b class="match term\d+">', '**', highlights)
                                highlights = highlights.replace('</b>', '**')
                                # 将连续的星号合并
                                highlights = re.sub(r'\*{2,}', '**', highlights)
                                # 移除可能残留的HTML标签
                                highlights = re.sub(r'<[^>]+>', '', highlights)
                                st.markdown(f"匹配内容: {highlights}")
                                st.write(f"文档内容片段: {result['content']}")
                                st.write("---")
                        else:
                            st.warning("没有找到相关文档")
                    except Exception as e:
                        st.error(f"搜索全文索时出错: {str(e)}")

        else:  # Neo4j 命令执行
            st.subheader("Neo4j 命令执行")
            with st.form(key='neo4j_query_form'):
                cypher_query = st.text_area("输入 Cypher 查询语句")
                submit_button = st.form_submit_button(label='执行 Neo4j 查询')
            if submit_button and cypher_query:
                with st.spinner("正在执行 Neo4j 查询..."):
                    try:
                        results = execute_neo4j_query(cypher_query)
                        if results:
                            st.success("查询执行成功！")
                            df = pd.DataFrame(results)
                            st.dataframe(df)
                        else:
                            st.info("查询执行成功，但没有返回结果。")
                    except Exception as e:
                        st.error(f"执行查询时发生错误: {str(e)}")

    with tab4:
        st.header("数据管理")

        # 全文索引信息
        if st.button("全文索引信息"):
            with st.spinner("正在获取全文索引信息..."):
                try:
                    ix = open_dir("fulltext_index")
                    with ix.searcher() as searcher:
                        all_docs = list(searcher.all_stored_fields())
                        st.write(f"全文索引中共有 {len(all_docs)} 个文档")
                        for doc in all_docs:
                            st.write(f"- 文件: {doc['title']}")
                except Exception as e:
                    st.error(f"获取全文索引信息时出错: {str(e)}")

        # 图数据库信息
        if st.button("图数据库信息"):
            with st.spinner("正在获取图数据库信息..."):
                try:
                    driver = get_neo4j_driver()
                    with driver.session() as session:
                        # 获取与上传文档相关的节点和关系
                        result = session.run("""
                        MATCH (n:Entity)
                        WHERE n.source IS NOT NULL
                        OPTIONAL MATCH (n)-[r:RELATED_TO]->(m:Entity)
                        WHERE m.source IS NOT NULL
                        RETURN n.name AS source, r.type AS relation, m.name AS target, n.source AS source_doc
                        """)

                        # 创建一个 NetworkX 图
                        G = nx.Graph()
                        for record in result:
                            source = record['source']
                            target = record['target']
                            relation = record['relation']
                            source_doc = record['source_doc']

                            # 添加节点
                            if source not in G:
                                G.add_node(source, title=f"来源: {source_doc}")
                            if target and target not in G:
                                G.add_node(target, title=f"来源: {source_doc}")

                            # 添加边
                            if target:
                                G.add_edge(source, target, title=relation)

                        # 创建 Pyvis 网络
                        net = Network(notebook=True, width="100%", height="600px", bgcolor="#222222", font_color="white")
                        net.from_nx(G)
                        net.toggle_physics(True)
                        net.show_buttons(filter_=['physics'])

                        # 保存并显示图
                        net.save_graph("graph.html")
                        with open("graph.html", "r", encoding="utf-8") as f:
                            graph_html = f.read()
                        st.components.v1.html(graph_html, width=None, height=600)

                        # 显示统计信息
                        st.write(f"图数据库中与上传文档相关的数据:")
                        st.write(f"- 节点数量: {G.number_of_nodes()}")
                        st.write(f"- 关系数量: {G.number_of_edges()}")

                except Exception as e:
                    st.error(f"获取图数据库信息时出错: {str(e)}")

        # 向量数据信息
        if st.button("向量数据信息"):
            with st.spinner("正在获取向量数据信息..."):
                try:
                    total_vectors = faiss_index.ntotal if faiss_index is not None else 0
                    st.write(f"向量数据库中共有 {total_vectors} 个向量")
                    st.write(f"向量维度: {faiss_index.d if faiss_index is not None else 'N/A'}")
                    st.write(f"已索引的文档数量: {len(st.session_state.file_indices)}")
                    
                    st.write("\n文件详细信息:")
                    for file_name, (chunks, index, patient_name) in st.session_state.file_indices.items():
                        st.write(f"- 文件: {file_name}")
                        st.write(f"  患者: {patient_name}")
                        st.write(f"  向量数量: {len(chunks)}")
                        st.write(f"  文本块数量: {len(chunks)}")
                    
                    if total_vectors != sum(len(chunks) for chunks, _, _ in st.session_state.file_indices.values()):
                        st.warning("注意：向量总数与文件索引中的向量数量不匹配，可能存在孤立向量。")
                    
                    if total_vectors == 0 and len(st.session_state.file_indices) == 0:
                        st.info("向量数据库当前为空。")
                except Exception as e:
                    st.error(f"获取向量数据信息时出错: {str(e)}")

        # 全文索引删除
        if st.button("全文索引删除"):
            with st.spinner("正在删除全文索引..."):
                try:
                    ix = open_dir("fulltext_index")
                    with ix.searcher() as searcher:
                        all_docs = list(searcher.all_stored_fields())
                        for doc in all_docs:
                            delete_fulltext_index(doc['title'])
                    st.success("全文索引已成功删除")
                except Exception as e:
                    st.error(f"删除全文索引时出错: {str(e)}")

        # 图数据删除
        if st.button("图数据删除"):
            with st.spinner("正在删除图数据..."):
                try:
                    driver = get_neo4j_driver()
                    with driver.session() as session:
                        # 只删除与上传文档相关的节点和关系
                        session.run("""
                        MATCH (n:Entity)
                        WHERE n.source IS NOT NULL
                        DETACH DELETE n
                        """)
                    st.success("与上传文档相关的图数据已成功删除")
                except Exception as e:
                    st.error(f"删除图数据时出错: {str(e)}")

        # 向量数据删除
        if st.button("向量数据删除"):
            with st.spinner("正在删除向量数据..."):
                try:
                    clear_vector_data()
                    st.session_state.file_indices = {}  # 清空文件索引
                    st.success("向量数据已成功删除")
                    st.rerun()
                except Exception as e:
                    st.error(f"删除向量数据时出错: {str(e)}")

        # 重新处理所有文档
        if st.button("重新处理所有文档"):
            with st.spinner("正在重新处理所有文档..."):
                try:
                    # 清除现有的向量数据
                    clear_vector_data()
                    st.session_state.file_indices = {}

                    # 重新处理所有文档
                    for file_name in os.listdir('indices'):
                        if file_name.endswith('.pkl'):
                            file_path = os.path.join('indices', file_name)
                            with open(file_path, 'rb') as f:
                                content = pickle.load(f)[0]  # 假设内容是第一个元素
                            
                            # 重新向量化文档
                            chunks, index, patient_name = vectorize_document(content, file_name[:-4], max_tokens)
                            st.session_state.file_indices[file_name[:-4]] = (chunks, index, patient_name)
                            save_index(file_name[:-4], chunks, index, patient_name)

                    st.success("所有文档已重新处理完成")
                    st.rerun()
                except Exception as e:
                    st.error(f"重新处理文档时出错: {str(e)}")

if __name__ == "__main__":
    main()
