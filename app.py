"""
AI知识问答系统
"""

import streamlit as st
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
import io  # 添加这一行
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from data_processor import (
    load_model, vectorize_document, extract_keywords, 
    search_documents, save_index, load_all_indices, 
    delete_index, rag_qa, initialize_openai,
    query_graph, hybrid_search, get_entity_relations,
    set_neo4j_config, get_neo4j_driver, process_data,
    generate_final_answer, vector_search, execute_neo4j_query,
    initialize_faiss  # 添加这一行
)

# 在文件顶部的导入语句之后添加
from data_processor import faiss_id_to_text, faiss_id_counter, faiss_index

# 设置页面配置
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
    api_key="sk-1pUmQlsIkgla3CuvKTgCrzDZ3r0pBxO608YJvIHCN18lvOrn",
    base_url="https://api.chatanywhere.tech/v1"
)

# 初始化 session state
if "file_indices" not in st.session_state:
    st.session_state.file_indices = load_all_indices()

def main():
    global faiss_id_to_text, faiss_id_counter, faiss_index
    
    st.title("AI知识问答系统")

    # 初始化 FAISS
    try:
        faiss_index = initialize_faiss()
        if faiss_index is None:
            st.error("FAISS 索引初始化失败。请检查 initialize_faiss() 函数。")
            return
    except Exception as e:
        st.error(f"FAISS 索引初始化时发生错误: {str(e)}")
        return

    # 加载所有索引
    st.session_state.file_indices = load_all_indices()
    
    # 如果有索引，将它们添加到 FAISS 索引中
    if st.session_state.file_indices:
        for file_name, (chunks, index) in st.session_state.file_indices.items():
            for i, chunk in enumerate(chunks):
                faiss_id_to_text[faiss_id_counter + i] = chunk
            vectors = index.reconstruct_n(0, index.ntotal)
            faiss_index.add(vectors)
        faiss_id_counter += sum(len(chunks) for chunks, _ in st.session_state.file_indices.values())

    # Neo4j 配置选择
    neo4j_option = st.radio(
        "选择 Neo4j 连接方式",
        ("Neo4j Aura", "本地 Neo4j")
    )

    if neo4j_option == "Neo4j Aura":
        CURRENT_NEO4J_CONFIG = set_neo4j_config("AURA")
        st.success("已选择并连接到 Neo4j Aura")
    else:
        CURRENT_NEO4J_CONFIG = set_neo4j_config("LOCAL")
        st.success("已选择并连接到本地 Neo4j")

    # 测试数据库连接
    if CURRENT_NEO4J_CONFIG:
        try:
            driver = get_neo4j_driver()
            with driver.session() as session:
                result = session.run("RETURN 1 AS test")
                test_value = result.single()["test"]
                if test_value == 1:
                    st.success("Neo4j 数据库连接成功")
                else:
                    st.error("Neo4j 数据库连接测试失败")
            driver.close()
        except Exception as e:
            st.error(f"连接到 Neo4j 数据库时出错: {str(e)}")
    else:
        st.error("Neo4j 配置无效或未设置")

    # 添加一个分隔线
    st.markdown("---")

    # 创建三个标签页
    tab1, tab2, tab3 = st.tabs(["文档上传", "知识库问答", "数据库检索"])

    with tab1:
        st.header("文档上传")
        
        # 设置最大token数
        max_tokens = 4096

        # 多文件上传
        uploaded_files = st.file_uploader("上传文档", type=["pdf", "docx", "txt"], accept_multiple_files=True)

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
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("加载到图数据库", key=f"graph_{uploaded_file.name}"):
                        with st.spinner(f"正在处理文档并加载到图数据库: {uploaded_file.name}..."):
                            result = process_data(content)
                        st.success(f"文档 {uploaded_file.name} 已成功加载到图数据库！")
                        st.write(f"处理了 {len(result['entities'])} 个实体和 {len(result['relations'])} 个关系")
                        
                        # 显示处理结果的详细信息
                        with st.expander("查看详细处理结果"):
                            st.subheader("实体:")
                            for entity in result['entities']:
                                st.write(f"- {entity}")
                            st.subheader("关系:")
                            for relation in result['relations']:
                                st.write(f"- {relation['source']} --[{relation['relation']}]--> {relation['target']}")
                
                with col2:
                    if st.button("加载到向量数据库", key=f"vector_{uploaded_file.name}"):
                        with st.spinner(f"正在处理文档并加载到向量数据库: {uploaded_file.name}..."):
                            chunks, index = vectorize_document(content, max_tokens)
                            st.session_state.file_indices[uploaded_file.name] = (chunks, index)
                            save_index(uploaded_file.name, chunks, index)
                        st.success(f"文档 {uploaded_file.name} 已成功加载到向量数据库！")
                        st.write(f"向 FAISS 向量数据库添加了 {len(chunks)} 个文本段落")

        # 显示已处理的文档并添加删除按钮
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

    with tab2:
        st.header("知识库问答")
        
        # 选择答类型
        qa_type = st.radio("选择问答类型", ["向量数据库问答", "图数据库问答", "混合问答"])
        
        if qa_type == "向量数据库问答":
            # 向量数据库问答部分
            st.subheader("向量数据库问答")
            vector_query = st.text_input("请输入您的问题（向量数据库）")
            if st.button("提交问题（向量）"):
                if vector_query:
                    with st.spinner("正在查询..."):
                        answer, sources, excerpt = rag_qa(vector_query, st.session_state.file_indices)
                    st.write("回答：", answer)
                    if sources:
                        st.write("参考来源：")
                        for source, _ in sources:
                            st.write(f"- {source}")
                    if excerpt:
                        st.write("相关原文：")
                        st.write(excerpt)
                else:
                    st.warning("请入问题")
        
        elif qa_type == "图数据库问答":
            # 图数据库问答部分
            st.subheader("图数据库问答")
            graph_query = st.text_input("请输入您的问题（图数据库）")
            if st.button("提交问题（图）"):
                if graph_query:
                    with st.spinner("正在��询..."):
                        answer = hybrid_search(graph_query)
                    st.write("回答：", answer)
                else:
                    st.warning("请输入问题")
        
        else:
            # 混合问答部分
            st.subheader("混合问答")
            hybrid_query = st.text_input("请输入您的问题（混合问答）")
            if st.button("提交问题（混合）"):
                if hybrid_query:
                    with st.spinner("正在查询..."):
                        # 图数据库查询
                        graph_answer = hybrid_search(hybrid_query)
                        
                        # 向量数据库查询
                        vector_answer, sources, excerpt = rag_qa(hybrid_query, st.session_state.file_indices)
                        
                        # 组合结果
                        combined_context = f"图数据库回答：{graph_answer}\n\n向量数据库回答：{vector_answer}"
                        if excerpt:
                            combined_context += f"\n\n相关原文：{excerpt}"
                        
                        # 使用大模型生成最终答案
                        final_answer = generate_final_answer(hybrid_query, combined_context)
                        
                        st.write("最终回答：", final_answer)
                        st.write("图数据库回答：", graph_answer)
                        st.write("向量数据库回答：", vector_answer)
                        if sources:
                            st.write("参考来源：")
                            for source, _ in sources:
                                st.write(f"- {source}")
                        if excerpt:
                            st.write("相关原文：")
                            st.write(excerpt)
                else:
                    st.warning("请输入问题")

        # 添加关键词搜索功能
        st.subheader("关键词搜索")
        search_keywords = st.text_input("输入关键词（用空格分隔）")
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

    with tab3:
        st.header("数据库检索")
        
        search_type = st.radio("选择搜索类型", ["图数据库搜索", "向量数据库搜索", "Neo4j 命令执行"])
        
        if search_type == "图数据库搜索":
            st.subheader("图数据库搜索")
            graph_query = st.text_input("输入搜索关键词")
            if st.button("执行图数据库搜索"):
                if graph_query:
                    with st.spinner("正在搜索图数据库..."):
                        entities, relations, contents = query_graph(graph_query)
                    if entities or relations:
                        st.success("搜索完成！")
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
                        net.save_graph("graph.html")
                        
                        with open("graph.html", 'r', encoding='utf-8') as f:
                            html_string = f.read()
                        components.html(html_string, height=600)
                    else:
                        st.warning("没有找到相关信息。")
                else:
                    st.warning("请输入搜索关键词。")

        elif search_type == "向量数据库搜索":
            st.subheader("向量数据库搜索")
            vector_query = st.text_input("输入搜索关键词")
            if st.button("执行向量数据库搜索"):
                if vector_query:
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
                else:
                    st.warning("请输入搜索关键词。")

        else:  # Neo4j 命令执行
            st.subheader("Neo4j 命令执行")
            cypher_query = st.text_area("输入 Cypher 查询语句")
            if st.button("执行 Neo4j 查询"):
                if cypher_query:
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
                else:
                    st.warning("请输入 Cypher 查询语句。")

if __name__ == "__main__":
    main()