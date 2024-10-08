import jieba
import jieba.posseg as pseg
import re
from neo4j import GraphDatabase
import logging
import openai
import json
from openai import OpenAI
import numpy as np
import faiss
import os
import tiktoken
from collections import Counter
from sentence_transformers import SentenceTransformer
import pickle
from PyPDF2 import PdfReader
import docx
import streamlit as st

# Neo4j连接配置
AURA_URI = "neo4j+s://b76a61f2.databases.neo4j.io:7687"
AURA_USERNAME = "neo4j"
AURA_PASSWORD = "JkVujA4SZWdifvfvj5m_gwdUgHsuTxQjbJQooUl1C14"

LOCAL_URI = "bolt://localhost:7687"
LOCAL_USERNAME = "test"
LOCAL_PASSWORD = "Mikeno01"

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CURRENT_NEO4J_CONFIG = {}

client = None

# 检查是否在 Streamlit 环境中运行
if 'streamlit' in globals():
    @st.cache_resource
    def load_model():
        return SentenceTransformer('all-MiniLM-L6-v2')
else:
    def load_model():
        return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# FAISS 相关的全局变量
faiss_index = None
faiss_id_to_text = {}
faiss_id_counter = 0

def set_neo4j_config(config_type):
    global CURRENT_NEO4J_CONFIG
    logger.info(f"设置 Neo4j 配置，类型: {config_type}")
    if config_type == "LOCAL":
        CURRENT_NEO4J_CONFIG = {
            "URI": LOCAL_URI,
            "USERNAME": LOCAL_USERNAME,
            "PASSWORD": LOCAL_PASSWORD
        }
        logger.info(f"设置本地 Neo4j 连接: {LOCAL_URI}")
    elif config_type == "AURA":
        CURRENT_NEO4J_CONFIG = {
            "URI": AURA_URI,
            "USERNAME": AURA_USERNAME,
            "PASSWORD": AURA_PASSWORD
        }
        logger.info(f"设置 Neo4j Aura 连接: {AURA_URI}")
    else:
        logger.error(f"未知的配置类型: {config_type}")
        return False
    
    if not all(CURRENT_NEO4J_CONFIG.values()):
        logger.error(f"Neo4j 配置不完整: {CURRENT_NEO4J_CONFIG}")
        return False
    
    logger.info(f"Neo4j 配置已设置: URI={CURRENT_NEO4J_CONFIG['URI']}, USERNAME={CURRENT_NEO4J_CONFIG['USERNAME']}")
    return CURRENT_NEO4J_CONFIG

def initialize_openai(api_key, base_url):
    global client
    client = OpenAI(api_key=api_key, base_url=base_url)
    logger.info("OpenAI 初始化完成")

def initialize_faiss():
    global faiss_index
    if faiss_index is None:
        try:
            faiss_index = faiss.IndexFlatL2(384)  # 384是向量维度,根据实际模型调整
            logger.info("FAISS 索引成功初始化")
        except Exception as e:
            logger.error(f"FAISS 索引初始化失败: {str(e)}")
            raise
    else:
        logger.info("FAISS 索引已经存在，跳过初始化")
    return faiss_index

# 计算token数量
def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(string))

# 文档向量化模块
def vectorize_document(content, max_tokens):
    chunks = []
    current_chunk = ""
    for sentence in content.split('.'):
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

# 提取关键词
def extract_keywords(text, top_k=5):
    words = jieba.cut(text)
    word_count = Counter(words)
    # 过滤掉停用词和单个字符
    keywords = [word for word, count in word_count.most_common(top_k*2) if len(word) > 1]
    return keywords[:top_k]

# 基于关键词搜文
def search_documents(keywords, file_indices):
    relevant_docs = []
    for file_name, (chunks, _) in file_indices.items():
        doc_content = ' '.join(chunks)
        if any(keyword in doc_content for keyword in keywords):
            relevant_docs.append(file_name)
    return relevant_docs

# 知识问答模块
def rag_qa(query, file_indices, relevant_docs=None):
    global client
    if client is None:
        raise ValueError("OpenAI client not initialized. Please call initialize_openai() first.")
    
    keywords = extract_keywords(query)
    if relevant_docs is None:
        relevant_docs = search_documents(keywords, file_indices)
    
    if not relevant_docs:
        return "没有找到相关文档。请尝试使用不同的关键。", [], ""

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
        if 0 <= i < len(all_chunks):  # 确保索引在有效范围内
            chunk = all_chunks[i]
            context.append(chunk)
            file_name = chunk_to_file.get(i, "未知文件")
            context_with_sources.append((file_name, chunk))

    context_text = "\n".join(context)
    
    # 确保总token数不超过4096
    max_context_tokens = 3000  # 为系统消息、查询和其他内容预留更多空间
    while num_tokens_from_string(context_text) > max_context_tokens:
        context_text = context_text[:int(len(context_text)*0.9)]  # 每减少10%的内容
    
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
    # 更新文件列表
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

def process_data(content):
    logger.info("开始处理数据")
    logger.info(f"接收到的内容: {content[:200]}...")  # 打印前200个字符

    # 改进的提示词
    prompt = f"""
    请从以下医疗记录中提取所有要的实体和关系。
    实体应包括但不限于：患者姓名、年龄、性别、诊断、症状、检查、治疗、药物、生理指标等。
    关系应描述实体之间的所有可能联系，如"患有"、"接受检查"、"使用药物"、"属性"等。
    请确保每个实体都至少有一个关系。对于没有明确关系的性（如性别、年龄等），请使用"属性"作为关系类型。
    请尽可能详细地取关系，不要遗任何可能的连接。
    请以JSON格式输出，格式如下：
    {{
        "entities": ["实体1", "实体2", ...],
        "relations": [
            {{"source": "实1", "relation": "关系", "target": "实体2"}},
            ...
        ]
    }}

    医疗记录：
    {content}
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "你是一个医疗信息提取助手，擅长医疗记录中提取体和关系。请尽可能详地提取所有相关信息。"},
            {"role": "user", "content": prompt}
        ]
    )

    result = response.choices[0].message.content
    logger.info(f"OpenAI API 返回的原始内容: {result}")

    try:
        # 尝试清理和解析JSON
        cleaned_result = re.search(r'\{.*\}', result, re.DOTALL)
        if cleaned_result:
            extracted_data = json.loads(cleaned_result.group())
        else:
            raise ValueError("无法在返回结果中找到有效的JSON")

        entities = extracted_data['entities']
        relations = extracted_data['relations']

        # 保所有实体都是字符串
        entities = [str(e) for e in entities]
        
        # 确保所有关系都是字典，且包含必要的键
        relations = [
            r if isinstance(r, dict) and all(k in r for k in ['source', 'relation', 'target'])
            else {'source': str(r[0]), 'relation': str(r[1]), 'target': str(r[2])}
            for r in relations
        ]

    except json.JSONDecodeError as e:
        logger.error(f"无法解析OpenAI返回的JSON: {str(e)}")
        # 使用则表达式提取实体���关系
        entities = re.findall(r'"([^"]+)"', result)
        relations = [
            {'source': m[0], 'relation': m[1], 'target': m[2]}
            for m in re.findall(r'\{"source": "([^"]+)", "relation": "([^"]+)", "target": "([^"]+)"\}', result)
        ]
    except Exception as e:
        logger.error(f"处理OpenAI返回结果时出错: {str(e)}")
        entities = []
        relations = []

    logger.info(f"提取的实体: {entities}")
    logger.info(f"提取的关系: {relations}")

    # 后处理逻辑
    patient_name = next((e for e in entities if "姓名" in e or "患者" in e), None)
    if patient_name:
        for entity in entities:
            if entity != patient_name and not any(r['source'] == patient_name and r['target'] == entity for r in relations):
                if entity in ["女", "男"]:
                    relations.append({"source": patient_name, "relation": "性别", "target": entity})
                elif entity.isdigit() or "岁" in entity:
                    relations.append({"source": patient_name, "relation": "年龄", "target": entity})
                elif entity in ["血糖", "血压", "体重", "心率", "体"]:
                    relations.append({"source": patient_name, "relation": "生理指���", "target": entity})
                elif entity in ["口干", "多尿", "多食", "体重下降"]:
                    relations.append({"source": patient_name, "relation": "症状", "target": entity})
                elif "检查" in entity or entity in ["心图", "胸片", "肌电图", "超声", "眼科检查", "GFR", "CGMS"]:
                    relations.append({"source": patient_name, "relation": "接受检查", "target": entity})
                elif "病" in entity or "症" in entity:
                    relations.append({"source": patient_name, "relation": "患有", "target": entity})
                elif "药" in entity or entity in ["胰岛素", "拜糖平", "二甲双胍", "诺和灵", "优泌灵"]:
                    relations.append({"source": patient_name, "relation": "使用药物", "target": entity})
                else:
                    relations.append({"source": patient_name, "relation": "相关", "target": entity})

    logger.info(f"处理后的实体: {entities}")
    logger.info(f"处理的关系: {relations}")

    driver = get_neo4j_driver()
    
    with driver.session() as session:
        # 创建实体
        for entity in entities:
            result = session.run("""
            MERGE (n:Entity {name: $name}) 
            SET n.content = $content
            RETURN count(*)
            """, name=str(entity).strip(), content=content)
            count = result.single()[0]
            logger.info(f"创建或更新实体: {entity}, : {count}")
        
        # 创建关系
        for relation in relations:
            if isinstance(relation, dict):
                source, rel_type, target = relation['source'], relation['relation'], relation['target']
            else:
                source, rel_type, target = relation
            result = session.run("""
            MATCH (a:Entity {name: $source})
            MERGE (b:Entity {name: $target})
            MERGE (a)-[r:RELATED_TO {type: $rel_type}]->(b)
            SET r.content = $content
            RETURN count(*)
            """, source=str(source).strip(), target=str(target).strip(), rel_type=str(rel_type).strip(), content=content)
            count = result.single()[0]
            logger.info(f"创建或更新关系: {source} -{rel_type}-> {target}, 影响的关系数: {count}")
    
    logger.info(f"处理了 {len(entities)} 个实体和 {len(relations)} 个关系")
    driver.close()
    return {"entities": entities, "relations": relations}

def query_graph(query):
    logger.info(f"执行图查询: {query}")
    driver = GraphDatabase.driver(
        CURRENT_NEO4J_CONFIG["URI"],
        auth=(CURRENT_NEO4J_CONFIG["USERNAME"], CURRENT_NEO4J_CONFIG["PASSWORD"])
    )
    
    with driver.session() as session:
        result = session.run("""
        CALL db.index.fulltext.queryNodes("entityFulltextIndex", $query) YIELD node, score
        OPTIONAL MATCH (node)-[r]-(m)
        RETURN node.name AS Entity, type(r) AS Relation, r.type AS RelationType, m.name AS Related, node.content AS Content, score
        ORDER BY score DESC
        LIMIT 10
        """, {"query": query})
        
        entities = set()
        relations = []
        contents = {}
        for record in result:
            if record["Entity"]:
                entities.add(record["Entity"])
                contents[record["Entity"]] = record["Content"]
            if record["Related"]:
                entities.add(record["Related"])
                relations.append({
                    "source": record["Entity"],
                    "relation": record["RelationType"] or record["Relation"],
                    "target": record["Related"]
                })
    
    driver.close()
    logger.info(f"查询结果 - 实体: {entities}")
    logger.info(f"查询结果 - 关系: {relations}")
    logger.info(f"查询结果 - 内容: {contents}")
    return list(entities), relations, contents

def hybrid_search(query):
    logger.info(f"执行混合搜索: {query}")
    try:
        entities, relations, contents = query_graph(query)
        
        if not entities and not relations:
            return "抱歉，我没有找到与您的问题相关的信息。请尝试用不同的方式问，或者确认所查询的信息是否已经录入系统。"
        
        # 限制上下文大小
        max_entities = 10
        max_relations = 20
        context = f"基于以下实体信息：\n"
        for entity in entities[:max_entities]:
            content = contents.get(entity, '无详细信息')
            if content is not None:
                context += f"{entity}: {content[:200]}...\n"
            else:
                context += f"{entity}: 无详细信息\n"
        context += "相关关系：\n"
        for relation in relations[:max_relations]:
            context += f"{relation['source']} {relation['relation']} {relation['target']}\n"
        
        prompt = f"{context}\n\n请根据上述信息回答问题：{query}\n\n回答："
        
        logger.info(f"发送到 OpenAI 的提示: {prompt}")
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个医疗助手，根据给定的实体信息和关系准确回答问题。如果信息不足，请如实说明。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        
        logger.info(f"OpenAI 响应: {response}")
        
        if response and response.choices and len(response.choices) > 0 and response.choices[0].message:
            answer = response.choices[0].message.content.strip()
            if not answer:
                answer = "抱歉，我无法根提供的信息回答这个问题。请尝试提供更多细节或以不同的方式提问。"
        else:
            answer = "抱歉，处理您的问题时出现了意外况。请稍后再试。"
        
        logger.info(f"搜索结果: {answer}")
        return answer
    except Exception as e:
        logger.error(f"混合搜索过程中发生错误: {str(e)}", exc_info=True)
        return f"抱歉，在处理您的问题时发生了错误: {str(e)}"

# 文件末尾添加以下函数

def query_graph_with_entities(entities):
    logger.info(f"执行图查询，实体: {entities}")
    driver = GraphDatabase.driver(
        CURRENT_NEO4J_CONFIG["URI"],
        auth=(CURRENT_NEO4J_CONFIG["USERNAME"], CURRENT_NEO4J_CONFIG["PASSWORD"])
    )
    
    try:
        with driver.session() as session:
            result = session.run("""
            MATCH (n:Entity)
            WHERE n.name IN $entities
            OPTIONAL MATCH (n)-[r]-(m)
            RETURN n.name AS Entity, type(r) AS Relation, r.type AS RelationType, m.name AS Related
            """, entities=entities)
            
            query_result = [{"Entity": record["Entity"], "Relation": record["Relation"], "RelationType": record["RelationType"], "Related": record["Related"]} for record in result]
            logger.info(f"图查询结果: {query_result}")
            return query_result
    except Exception as e:
        logger.error(f"图查询过程中发生错误: {str(e)}")
        return []
    finally:
        driver.close()

def get_entity_relations(entity_name):
    logger.info(f"查询实体 {entity_name} 的相关信息")
    driver = GraphDatabase.driver(
        CURRENT_NEO4J_CONFIG["URI"],
        auth=(CURRENT_NEO4J_CONFIG["USERNAME"], CURRENT_NEO4J_CONFIG["PASSWORD"])
    )
    
    try:
        with driver.session() as session:
            result = session.run("""
            MATCH (n:Entity {name: $name})
            OPTIONAL MATCH (n)-[r]-(m)
            RETURN n.name AS Entity, type(r) AS Relation, r.type AS RelationType, m.name AS Related
            """, name=entity_name)
            
            query_result = [{"Entity": record["Entity"], "Relation": record["Relation"], "RelationType": record["RelationType"], "Related": record["Related"]} for record in result]
            logger.info(f"实体 {entity_name} 的询结果: {query_result}")
            return query_result
    except Exception as e:
        logger.error(f"查询实体 {entity_name} 时发生错误: {str(e)}")
        return []
    finally:
        driver.close()

# 在文件的适当位置添加这个函数，比如在 set_neo4j_config 函数之后
def get_neo4j_driver():
    return GraphDatabase.driver(
        CURRENT_NEO4J_CONFIG["URI"],
        auth=(CURRENT_NEO4J_CONFIG["USERNAME"], CURRENT_NEO4J_CONFIG["PASSWORD"])
    )

def generate_final_answer(query, combined_context):
    prompt = f"""
    基于以下来自图数据库和向量数据库的信息，请回答问题。如果两个数据库的信息有冲突，请综合分析并给出最合理的回答。
    如果信息不足以回答问题，请如实说明。

    问题：{query}

    信息：
    {combined_context}

    请提供一个综合的回答：
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "你是一个智能助手，能够综合分析来自不同数据源的信息，并提供准确、全面的回答。"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )
    
    return response.choices[0].message.content.strip()

def vector_search(query, k=5):
    global faiss_index
    if faiss_index is None:
        logger.error("FAISS 索引未初始化")
        return []
    
    query_vector = model.encode([query])
    D, I = faiss_index.search(query_vector.astype('float32'), k)
    
    results = []
    for i, d in zip(I[0], D[0]):
        if i != -1 and i in faiss_id_to_text:
            results.append({"text": faiss_id_to_text[i], "distance": float(d)})
    
    return results

def execute_neo4j_query(cypher_query):
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run(cypher_query)
        return [record.data() for record in result]

# 定义 __all__ 列表
__all__ = [
    'set_neo4j_config',
    'initialize_openai',
    'process_data',
    'query_graph',
    'hybrid_search',
    'CURRENT_NEO4J_CONFIG',
    'query_graph_with_entities',
    'get_entity_relations',
    'initialize_faiss',  # 确保这一行存在
    'vectorize_document',
    'rag_qa',
    'save_index',
    'load_all_indices',
    'delete_index',
    'extract_keywords',
    'search_documents',
    'get_neo4j_driver',
    'generate_final_answer',
    'vector_search',
    'execute_neo4j_query',
    'faiss_index',
    'faiss_id_to_text',
    'faiss_id_counter'
]