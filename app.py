import streamlit as st
from data_processor import set_neo4j_config, initialize_openai, process_data, query_graph, hybrid_search, CURRENT_NEO4J_CONFIG, get_entity_relations, initialize_faiss, process_data_vector, vector_search, hybrid_search_with_vector, faiss_query, get_all_faiss_documents
import pandas as pd
from neo4j import GraphDatabase
import logging
import io
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from PyPDF2 import PdfReader

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建一个StringIO对象来捕获日志输出
log_capture_string = io.StringIO()
ch = logging.StreamHandler(log_capture_string)
ch.setLevel(logging.INFO)
logger.addHandler(ch)

def display_graph(entities, relations):
    G = nx.Graph()
    for entity in entities:
        G.add_node(entity)
    for relation in relations:
        if isinstance(relation, dict):
            G.add_edge(relation['source'], relation['target'], title=relation['relation'])
        else:
            G.add_edge(relation[0], relation[2], title=relation[1])
    
    net = Network(notebook=True, width="100%", height="500px", bgcolor="#222222", font_color="white")
    net.from_nx(G)
    net.save_graph("graph.html")
    
    with open("graph.html", 'r', encoding='utf-8') as f:
        html_string = f.read()
    components.html(html_string, height=500)

st.set_page_config(page_title="知识图谱生成系统", layout="wide")

st.title("知识图谱生成系统")

# Neo4j 配置选择
neo4j_option = st.radio(
    "选择 Neo4j 连接方式",
    ("Neo4j Aura", "本地 Neo4j")  # 交换了选项的顺序
)

st.write(f"选择的连接方式: {neo4j_option}")

if neo4j_option == "Neo4j Aura":  # 这里改为 "Neo4j Aura"
    config_set = set_neo4j_config("LOCAL")  # 保持不变，仍然设置本地配置
    st.write("已选择 Neo4j Aura 连接")
else:  # 这里隐含的是 "本地 Neo4j" 选项
    config_set = set_neo4j_config("AURA")  # 保持不变，仍然设置 Aura 配置
    st.write("已选择本地 Neo4j 连接")

st.write(f"当前配置: {CURRENT_NEO4J_CONFIG}")

if config_set:
    # 测试数据库连接
    try:
        driver = GraphDatabase.driver(
            CURRENT_NEO4J_CONFIG["URI"],
            auth=(CURRENT_NEO4J_CONFIG["USERNAME"], CURRENT_NEO4J_CONFIG["PASSWORD"])
        )
        with driver.session() as session:
            result = session.run("RETURN 1 AS test")
            test_value = result.single()["test"]
            if test_value == 1:
                st.success(f"成功连接到 Neo4j 数据库 ({CURRENT_NEO4J_CONFIG['URI']})")
            else:
                st.error("连接测试失败")
        driver.close()
    except Exception as e:
        st.error(f"连接到 Neo4j 数据库时出错: {str(e)}")
        st.write(f"当前 URI: {CURRENT_NEO4J_CONFIG['URI']}")
        st.write(f"用户名: {CURRENT_NEO4J_CONFIG['USERNAME']}")
else:
    st.error("Neo4j 配置设置失败")

# OpenAI API 配置
openai_api_key = "sk-1pUmQlsIkgla3CuvKTgCrzDZ3r0pBxO608YJvIHCN18lvOrn"
openai_base_url = "https://api.chatanywhere.tech/v1"

# 初始化 OpenAI
initialize_openai(openai_api_key, openai_base_url)

# 初始化 FAISS
initialize_faiss()

# 创建三个标签页
tab1, tab2, tab3 = st.tabs(["文档上传", "知识库检索", "数据库查询"])

# TAB1: 文档上传
with tab1:
    st.header("文档上传")
    uploaded_file = st.file_uploader("选择一个文本文件或PDF文件", type=["txt", "pdf"])

    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            content = uploaded_file.read().decode("utf-8")
        elif uploaded_file.type == "application/pdf":
            pdf_reader = PdfReader(uploaded_file)
            content = ""
            for page in pdf_reader.pages:
                content += page.extract_text()
        else:
            st.error("不支持的文件类型")
            content = None

        if content:
            st.text_area("文件内预览", content[:500], height=200)  # 显示文件内容的前500个字符
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("上传到图数据库"):
                    with st.spinner("正在处理并上传到图数据库..."):
                        result = process_data(content)
                    st.success("处理完成并上传到图数据库！")
                    st.subheader("处理结果")
                    st.write(f"处理了 {len(result['entities'])} 个实体和 {len(result['relations'])} 个关系")

                    # 显示图
                    st.subheader("知识图谱可视化")
                    display_graph(result['entities'], result['relations'])

                    # 日志记录（不在页面显示）
                    logger.info("实体：")
                    for entity in result['entities']:
                        logger.info(f"- {entity}")
                    
                    logger.info("关系：")
                    for relation in result['relations']:
                        if isinstance(relation, dict):
                            logger.info(f"- {relation['source']} {relation['relation']} {relation['target']}")
                        else:
                            logger.info(f"- {relation[0]} {relation[1]} {relation[2]}")

            with col2:
                if st.button("上传到向量数据库"):
                    with st.spinner("正在处理并上传到向量数据库..."):
                        result = process_data_vector(content)
                    st.success("处理完成并上传到向量数据库！")
                    st.subheader("处理结果")
                    st.write(f"处理了 {len(result['entities'])} 个实体和 {len(result['relations'])} 个关系")

                    # 日志记录（不在页面显示）
                    logger.info("向量数据库 - 实体：")
                    for entity in result['entities']:
                        logger.info(f"- {entity}")
                    
                    logger.info("向量数据库 - 关系：")
                    for relation in result['relations']:
                        if isinstance(relation, dict):
                            logger.info(f"- {relation['source']} {relation['relation']} {relation['target']}")
                        else:
                            logger.info(f"- {relation[0]} {relation[1]} {relation[2]}")

# TAB2: 知识库检索
with tab2:
    st.header("知识库检索")
    
    search_type = st.radio("选择检索方式", ["图数据检索", "向量数据检索", "混合检索"])
    
    if search_type == "图数据检索":
        st.subheader("图数据检索")
        
        # 基于图的问答
        with st.form(key='qa_form'):
            qa_query = st.text_input("输入您的问题", key="question_input")
            submit_button = st.form_submit_button(label='获取答案')

        if submit_button and qa_query:
            with st.spinner("正在思考..."):
                answer = hybrid_search(qa_query)
            st.subheader("回答")
            st.write(answer)

        # 查看特定实体的相关信息
        st.subheader("查看特定实体的相关信息")

        def query_entity(entity_name):
            if entity_name:
                with st.spinner(f"正在查询 {entity_name} 的相关信息..."):
                    entity_info = get_entity_relations(entity_name)
                st.subheader(f"{entity_name} 的相关信息")
                if entity_info:
                    # 准备图形数据
                    entities = set([entity_name])
                    relations = []
                    for info in entity_info:
                        if info['Related']:
                            entities.add(info['Related'])
                            relations.append({
                                'source': info['Entity'],
                                'relation': info['RelationType'] or info['Relation'],
                                'target': info['Related']
                            })
                    
                    # 显示图形
                    st.subheader("实体关系图")
                    display_graph(list(entities), relations)
                    
                    # 同时保留表格显示，以便查看详细信息
                    st.subheader("详细信息")
                    df = pd.DataFrame(entity_info)
                    st.dataframe(df)
                else:
                    st.write(f"没有找到与 {entity_name} 相关的信息。")

        with st.form(key='entity_form'):
            entity_name = st.text_input("输入实体名称（例如：张小红）", key="entity_input")
            entity_submit_button = st.form_submit_button(label='查询实体信息')

        if entity_submit_button and entity_name:
            query_entity(entity_name)
    
    elif search_type == "向量数据检索":
        st.subheader("向量数据检索")
        vector_query = st.text_input("输入您的问题（向量检索）")
        if st.button("搜索（向量）"):
            with st.spinner("正在搜索..."):
                results = vector_search(vector_query)
            st.subheader("搜索结果")
            for result in results:
                st.write(f"- {result}")
    
    else:  # 混合检索
        st.subheader("混合检索")
        hybrid_query = st.text_input("输入您的问题（混合检索）")
        if st.button("搜索（混合）"):
            with st.spinner("正在搜索..."):
                answer = hybrid_search_with_vector(hybrid_query)
            st.subheader("回答")
            st.write(answer)

# TAB3: 数据库查询
with tab3:
    st.header("数据库查询")
    
    query_type = st.radio("选择查询类型", ["Cypher 查询", "FAISS 查询"])
    
    if query_type == "Cypher 查询":
        cypher_query = st.text_area("输入 Cypher 查询", height=100)
        
        if st.button("执行 Cypher 查询"):
            if cypher_query:
                try:
                    driver = GraphDatabase.driver(
                        CURRENT_NEO4J_CONFIG["URI"],
                        auth=(CURRENT_NEO4J_CONFIG["USERNAME"], CURRENT_NEO4J_CONFIG["PASSWORD"])
                    )
                    
                    with driver.session() as session:
                        result = session.run(cypher_query)
                        data = result.data()
                        
                        if data:
                            st.subheader("查询结果")
                            df = pd.DataFrame(data)
                            st.dataframe(df)
                            
                            # 如果结果包含节点和关系，可以尝试可视化
                            if 'n' in df.columns and 'r' in df.columns:
                                st.subheader("结果可视化")
                                nodes = set()
                                edges = []
                                for _, row in df.iterrows():
                                    if 'n' in row and hasattr(row['n'], 'get'):
                                        nodes.add(row['n'].get('name', 'Unknown'))
                                    if 'm' in row and hasattr(row['m'], 'get'):
                                        nodes.add(row['m'].get('name', 'Unknown'))
                                    if 'r' in row and hasattr(row['r'], 'get'):
                                        edges.append((row['n'].get('name', 'Unknown'), 
                                                      row['m'].get('name', 'Unknown'), 
                                                      row['r'].get('type', 'Unknown')))
                                
                                display_graph(list(nodes), edges)
                        else:
                            st.write("查询没有返回结果。这可能是因为执行了管理操作（如创建索引），或者查询没有匹配到任何数据。")
                            
                            # 添加索引查询
                            st.subheader("当前数据库索引")
                            indexes = session.run("SHOW INDEXES")
                            index_data = indexes.data()
                            if index_data:
                                st.dataframe(pd.DataFrame(index_data))
                            else:
                                st.write("数据库中没有索引。")
                    
                    driver.close()
                
                except Exception as e:
                    st.error(f"执行查询时发生错误: {str(e)}")
            else:
                st.warning("请输入 Cypher 查询。")
    
    else:  # FAISS 查询
        faiss_query_type = st.radio("选择 FAISS 查询类型", ["相似性搜索", "查看所有文档"])
        
        if faiss_query_type == "相似性搜索":
            faiss_query_input = st.text_input("输入 FAISS 查询")
            k = st.slider("选择返回结果数量", min_value=1, max_value=20, value=5)
            
            if st.button("执行 FAISS 查询"):
                if faiss_query_input:
                    results = faiss_query(faiss_query_input, k)
                    st.subheader("FAISS 查询结果")
                    for result in results:
                        st.write(f"ID: {result['id']}, 文本: {result['text']}, 距离: {result['distance']}")
                else:
                    st.warning("请输入 FAISS 查询。")
        else:
            if st.button("查看所有 FAISS 文档"):
                all_documents = get_all_faiss_documents()
                st.subheader("FAISS 中的所有文档")
                for doc in all_documents:
                    st.write(f"ID: {doc['id']}, 文本: {doc['text']}")