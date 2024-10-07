import streamlit as st
from data_processor import set_neo4j_config, initialize_openai, process_data, query_graph, hybrid_search, CURRENT_NEO4J_CONFIG, get_entity_relations
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
    ("本地 Neo4j", "Neo4j Aura")
)

if neo4j_option == "本地 Neo4j":
    set_neo4j_config("LOCAL_URI", "LOCAL_USERNAME", "LOCAL_PASSWORD")
else:
    set_neo4j_config("AURA_URI", "AURA_USERNAME", "AURA_PASSWORD")

# OpenAI API 配置
openai_api_key = "sk-1pUmQlsIkgla3CuvKTgCrzDZ3r0pBxO608YJvIHCN18lvOrn"
openai_base_url = "https://api.chatanywhere.tech/v1"

# 初始化 OpenAI
initialize_openai(openai_api_key, openai_base_url)

# 文件上传和处理
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
        st.text_area("文件内容预览", content[:500], height=200)  # 显示文件内容的前500个字符
        
        if st.button("处理文件并生成知识图谱"):
            with st.spinner("正在处理..."):
                result = process_data(content)
            st.success("处理完成！")
            st.subheader("处理结果")
            st.write(f"处理了 {len(result['entities'])} 个实体和 {len(result['relations'])} 个关系")
            
            # 显示实体
            st.write("实体：")
            for entity in result['entities']:
                st.write(f"- {entity}")
            
            # 显示关系
            st.write("关系：")
            for relation in result['relations']:
                if isinstance(relation, dict):
                    st.write(f"- {relation['source']} {relation['relation']} {relation['target']}")
                else:
                    st.write(f"- {relation[0]} {relation[1]} {relation[2]}")

            # 显示图
            st.subheader("知识图谱可视化")
            display_graph(result['entities'], result['relations'])

            # 在处理完成后添加这段代码
            st.write("处理摘要：")
            st.write(f"- 总共处理了 {len(result['entities'])} 个实体")
            st.write(f"- 总共处理了 {len(result['relations'])} 个关系")
            st.write(f"- 主要实体包括：{', '.join(result['entities'][:5])}...")

            # 清空日志缓存，为下一次处理做准备
            log_capture_string.truncate(0)
            log_capture_string.seek(0)

# 基于图的问答
st.header("基于图的问答")

with st.form(key='qa_form'):
    qa_query = st.text_input("输入您的问题", key="question_input")
    submit_button = st.form_submit_button(label='获取答案')

if submit_button and qa_query:
    with st.spinner("正在思考..."):
        answer = hybrid_search(qa_query)
    st.subheader("回答")
    st.write(answer)

# 查看特定实体的相关信息
st.header("查看特定实体的相关信息")

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