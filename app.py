"""
基于图数据库及向量数据库的混合问答系统
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
    open_dir, delete_graph_data, delete_vector_data, delete_fulltext_index
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

def main():
    if 'faiss_id_to_text' not in st.session_state:
        st.session_state.faiss_id_to_text = {}
    if 'faiss_id_counter' not in st.session_state:
        st.session_state.faiss_id_counter = 0
    
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
                st.session_state.faiss_id_to_text[st.session_state.faiss_id_counter + i] = chunk
            vectors = index.reconstruct_n(0, index.ntotal)
            faiss_index.add(vectors)
        st.session_state.faiss_id_counter += sum(len(chunks) for chunks, _ in st.session_state.file_indices.values())

    # Neo4j 配置选择
    col1, col2, col3 = st.columns(3)
    with col1:
        neo4j_option = st.radio(
            "选择 Neo4j 连接方式",
            ("Neo4j Aura", "本地 Neo4j")
        )

    with col2:
        if neo4j_option == "Neo4j Aura":
            CURRENT_NEO4J_CONFIG = set_neo4j_config("AURA")
            connection_status = "已选择连接到 Neo4j Aura"
        else:
            CURRENT_NEO4J_CONFIG = set_neo4j_config("LOCAL")
            connection_status = "已选择连接到本地 Neo4j"

    with col3:
        # 测试数据库连接
        if CURRENT_NEO4J_CONFIG:
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
        else:
            connection_status += " - 配置无效或未设置"

    st.write(connection_status)

    # 添加一个分隔线
    st.markdown("---")

    # 创建四个标签页
    tab1, tab2, tab3, tab4 = st.tabs(["文档上传", "知识库问答", "数据库检索", "数据管理"])

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
                
                col1, col2, col3 = st.columns(3)
                
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
                
                with col3:
                    if st.button("创建全文索引", key=f"fulltext_{uploaded_file.name}"):
                        with st.spinner(f"正在为文档创建全文索引: {uploaded_file.name}..."):
                            try:
                                create_fulltext_index(content, uploaded_file.name)
                                st.success(f"文档 {uploaded_file.name} 已成功创建全文索引！")
                                # 添加验证步骤
                                ix = open_dir("fulltext_index")
                                with ix.searcher() as searcher:
                                    results = searcher.search(QueryParser("title", ix.schema).parse(uploaded_file.name))
                                    if results:
                                        st.success(f"成功验证文档 {uploaded_file.name} 已被索引")
                                        st.info(f"索引中的文档内容长度: {sum(len(hit['content']) for hit in results)}")
                                        st.info(f"索引中的第一个文档块内容前200字符: {results[0]['content'][:200]}")
                                    else:
                                        st.warning(f"无法在索引中找到文档 {uploaded_file.name}")
                            except Exception as e:
                                st.error(f"创建或验证索引时出错: {str(e)}")
                                st.error(f"错误类型: {type(e).__name__}")
                                st.error(f"错误详情: {e.args}")

        # 显示已处理的文档并添加删除按钮
        st.subheader("已处理文档:")
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
                            
                            # 删除全文索引中的数据
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
        
        qa_type = st.radio("选择问答类型", ["向量数据库问答", "图数据库问答", "混合问答"])
        
        if qa_type == "向量数据库问答":
            st.subheader("向量数据库问答")
            with st.form(key='vector_qa_form'):
                vector_query = st.text_input("请输入您的问题（向量数据库）")
                submit_button = st.form_submit_button(label='提交问题')
            if submit_button and vector_query:
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
        
        elif qa_type == "图数据库问答":
            st.subheader("图数据库问答")
            with st.form(key='graph_qa_form'):
                graph_query = st.text_input("请输入您的题（图数据库）")
                submit_button = st.form_submit_button(label='提交问题')
            if submit_button and graph_query:
                with st.spinner("正在查询..."):
                    answer = hybrid_search(graph_query)
                st.write("回答：", answer)
        
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
                        st.write(f"全文索引中共有 {len(fulltext_results)} 个相关文档被搜索")
                    except Exception as e:
                        st.error(f"全文检索出错: {str(e)}")
                        fulltext_results = []

                    # 图数据库查询
                    graph_answer, graph_entities, graph_relations = hybrid_search(hybrid_query)
                    
                    # 向量数据库查
                    vector_answer, sources, excerpt = rag_qa(hybrid_query, st.session_state.file_indices)
                    
                    # 使用所有结果成最终答案
                    final_answer = generate_final_answer(
                        hybrid_query, 
                        graph_answer, 
                        vector_answer, 
                        fulltext_results,  # 确保这里传递了 fulltext_results
                        excerpt, 
                        graph_entities, 
                        graph_relations
                    )
                    
                    st.write("最终回答：", final_answer)
                    st.write("图数据库回答：", graph_answer)
                    st.write("向量数据库回答：", vector_answer)
                    st.write(f"全文检索结果数量：{len(fulltext_results)}")

                    # 显示全文检结果
                    if fulltext_results:
                        st.write("全文检索结果（前3个）：")
                        for result in fulltext_results[:3]:
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
                            st.write(f"  文档内容片段: {result['content']}")
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
        
        search_type = st.radio("选择搜索类型", ["图数据库搜索", "向量数据库搜索", "全文索引搜索", "Neo4j 命令执行"])
        
        if search_type == "图数据库搜索":
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
                                st.write(f"文档: {result['title']}, 相关度: {result['score']:.2f}")
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
                        st.error(f"搜索全文索引时出错: {str(e)}")

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

        col1, col2 = st.columns(2)

        with col1:
            # 全文索引信息
            if st.button("全文索引信息"):
                with st.spinner("正在获取全文索引信息..."):
                    try:
                        ix = open_dir("fulltext_index")
                        with ix.searcher() as searcher:
                            all_docs = list(searcher.all_stored_fields())
                            st.write(f"全文索引中共有 {len(all_docs)} 个文档")
                            for doc in all_docs:
                                st.write(f"- 文档: {doc['title']}")
                    except Exception as e:
                        st.error(f"获取全文索引信息时出错: {str(e)}")

            # 图数据库信息
            if st.button("图数据库信息"):
                with st.spinner("正在获取图数据库信息..."):
                    try:
                        driver = get_neo4j_driver()
                        with driver.session() as session:
                            # 获取与上传文档相关的节点数量
                            result = session.run("""
                            MATCH (n:Entity)
                            WHERE n.source IS NOT NULL
                            RETURN count(n) as node_count
                            """)
                            node_count = result.single()["node_count"]

                            # 获取与上传文档相关的关系数量
                            result = session.run("""
                            MATCH (n:Entity)-[r]->(m:Entity)
                            WHERE n.source IS NOT NULL AND m.source IS NOT NULL
                            RETURN count(r) as relation_count
                            """)
                            relation_count = result.single()["relation_count"]

                            st.write(f"图数据库中与上传文档相关的数据:")
                            st.write(f"- 节点数量: {node_count}")
                            st.write(f"- 关系数量: {relation_count}")

                            # 获取一些示例节点和关系
                            result = session.run("""
                            MATCH (n:Entity)
                            WHERE n.source IS NOT NULL
                            RETURN n.name AS name, n.source AS source
                            LIMIT 5
                            """)
                            st.write("示例节点:")
                            for record in result:
                                st.write(f"  - {record['name']} (来源: {record['source']})")

                            result = session.run("""
                            MATCH (n:Entity)-[r]->(m:Entity)
                            WHERE n.source IS NOT NULL AND m.source IS NOT NULL
                            RETURN n.name AS source, type(r) AS relation, m.name AS target
                            LIMIT 5
                            """)
                            st.write("示例关系:")
                            for record in result:
                                st.write(f"  - {record['source']} --[{record['relation']}]--> {record['target']}")

                    except Exception as e:
                        st.error(f"获取图数据库信息时出错: {str(e)}")

            # 向量数据信息
            if st.button("向量数据信息"):
                with st.spinner("正在获取向量数据信息..."):
                    try:
                        st.write(f"向量数据库中共有 {faiss_index.ntotal} 个向量")
                        st.write(f"向量维度: {faiss_index.d}")
                        st.write(f"已索引的文档数量: {len(st.session_state.file_indices)}")
                    except Exception as e:
                        st.error(f"获取向量数据信息时出错: {str(e)}")

        with col2:
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
            if st.button("数据删除"):
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
                        # 重新初始化 FAISS 索引
                        initialize_faiss()
                        st.session_state.faiss_id_to_text = {}
                        st.session_state.faiss_id_counter = 0
                        st.session_state.file_indices = {}  # 清空文件索引
                        st.success("向量数据已成功删除")
                    except Exception as e:
                        st.error(f"删除向量数据时出错: {str(e)}")

if __name__ == "__main__":
    main()