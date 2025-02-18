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
from transformers import AutoTokenizer, AutoModel
import torch
from whoosh.index import create_in, open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser, OrGroup, AndGroup, MultifieldParser
from whoosh.query import And, Or, Term
from whoosh.analysis import Analyzer, Token
from gensim.models import KeyedVectors
from whoosh.analysis import Analyzer, Token
from whoosh.fields import Schema, TEXT, ID
from whoosh.index import create_in, open_dir
from whoosh.qparser import QueryParser
from whoosh.searching import NoTermsException

# 在文件开头声明全局变量
global CURRENT_NEO4J_CONFIG

# Neo4j连接配置
AURA_URI = "neo4j+s://85c689ad.databases.neo4j.io:7687"
AURA_USERNAME = "neo4j"
AURA_PASSWORD = "XL37Q-0UhF1YA2diY3f9Ah3dLxHmWlyoN6rexDu9sdA"

LOCAL_URI = "bolt://localhost:7687"
LOCAL_USERNAME = "test"
LOCAL_PASSWORD = "Mikeno01"

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化默认配置
CURRENT_NEO4J_CONFIG = {
    "URI": AURA_URI,
    "USERNAME": AURA_USERNAME,
    "PASSWORD": AURA_PASSWORD
}

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
        logger.error(f"Neo4j 配置不完: {CURRENT_NEO4J_CONFIG}")
        return False
    
    logger.info(f"Neo4j 配已设置: URI={CURRENT_NEO4J_CONFIG['URI']}, USERNAME={CURRENT_NEO4J_CONFIG['USERNAME']}")
    return CURRENT_NEO4J_CONFIG

def initialize_openai():
    return OpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY", st.secrets["openrouter"]["api_key"]),
        base_url=os.environ.get("OPENROUTER_BASE_URL", st.secrets["openrouter"]["base_url"]),
        timeout=60.0
    )

def initialize_faiss():
    global faiss_index, faiss_id_to_text, faiss_id_counter
    faiss_index = faiss.IndexFlatL2(384)  # 384是向量维度,根据实际模型调整
    faiss_id_to_text = {}
    faiss_id_counter = 0
    logger.info("FAISS 索引已重新初始化")
    return faiss_index

# 计token数量
def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(string))

# 文档向量化模块
def vectorize_document(content, file_name, max_tokens):
    # 从文件名中提取患者姓名（去掉.pdf后缀）
    patient_name = os.path.splitext(file_name)[0]
    logger.info(f"从文件名 {file_name} 中提取的患者姓名: {patient_name}")
    
    chunks = []
    current_chunk = ""
    sentences = content.split('。')
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > max_tokens:
            if current_chunk:
                chunks.append(f"患者：{patient_name}。{current_chunk}")
            current_chunk = sentence
        else:
            current_chunk += sentence + "。"
    
    if current_chunk:
        chunks.append(f"患者：{patient_name}。{current_chunk}")
    
    vectors = model.encode(chunks)
    index = faiss.IndexFlatL2(384)  # 384是向量维度,根据实际模型调整
    index.add(vectors)
    
    # 同步到全局 faiss_index
    global faiss_index
    if faiss_index is None:
        faiss_index = faiss.IndexFlatL2(384)
    faiss_index.add(vectors)
    
    logger.info(f"文件 {file_name} 已被分成 {len(chunks)} 个文本块")
    return chunks, index, patient_name

# 添加提取患者姓名的函数
def extract_patient_name(file_name):
    # 直接从文件名提取患者姓名（去掉.pdf后缀）
    return os.path.splitext(file_name)[0]

# 提取关键词
def extract_keywords(text, top_k=5):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    model = AutoModel.from_pretrained("bert-base-chinese")
    
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(**inputs)
    
    # 使用最后一层的隐藏状态
    last_hidden_state = outputs.last_hidden_state[0]
    
    # 计算每个词的重要得分（这里使用简单的L2数）
    word_importance = torch.norm(last_hidden_state, dim=1)
    
    # 获取top_k个重要的词
    top_k_indices = torch.argsort(word_importance, descending=True)[:top_k]
    top_k_tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0][top_k_indices])
    
    return top_k_tokens

# 基于关键词搜文
def search_documents(keywords, file_indices):
    relevant_docs = []
    for file_name, (chunks, _, _) in file_indices.items():
        doc_content = ' '.join(chunks)
        if any(keyword in doc_content for keyword in keywords):
            relevant_docs.append(file_name)
    return relevant_docs

# 知识问答模块
def rag_qa(query, file_indices, k=10):
    logger.info(f"执行 RAG 问答: {query}")
    
    # 向量化查询
    query_vector = model.encode([query])
    
    # 在所有文档中搜索最相关的文本块
    all_results = []
    for file_name, (chunks, index, patient_name) in file_indices.items():
        D, I = index.search(query_vector, k)
        for i, (dist, idx) in enumerate(zip(D[0], I[0])):
            if idx != -1:
                all_results.append({
                    "text": chunks[idx],
                    "distance": dist,
                    "file_name": file_name,
                    "patient_name": patient_name
                })
    
    # 按相似度排序并选择前k个结果
    all_results.sort(key=lambda x: x["distance"])
    top_results = all_results[:k]
    
    # 构建提示
    context = "\n\n".join([f"文件: {r['file_name']}\n患者: {r['patient_name']}\n内容: {r['text']}" for r in top_results])
    
    prompt = f"""基于以下信息回答问题。如果信息不足以回答问题，请如实说明。

问题：{query}

相关信息：
{context}

请提供详细的回答，并在回答后附上最相关的原文摘录，以"相关原文："为前缀。确保回答中提到的患者信息与上下文中的患者信息一致。
"""

    # 发送给大模型
    response = client.chat.completions.create(
        model=get_model_name(),
        messages=[
            {"role": "system", "content": "你是一个医疗助手，根据给定的病历信息回答问题。请确保回答准确、相关，并引用原文。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        top_p=0.8,
        max_tokens=2048
    )
    
    answer = response.choices[0].message.content.strip()
    
    # 解析回答和相关原文
    if "相关原文：" in answer:
        main_answer, relevant_excerpt = answer.split("相关原文：", 1)
    else:
        main_answer, relevant_excerpt = answer, ""
    
    return main_answer.strip(), top_results, relevant_excerpt.strip()

# 保存索引和chunks
def save_index(file_name, chunks, index, patient_name):
    if not os.path.exists('indices'):
        os.makedirs('indices')
    with open(f'indices/{file_name}.pkl', 'wb') as f:
        pickle.dump((chunks, index, patient_name), f)
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
                    chunks, index, patient_name = pickle.load(f)
                file_indices[file_name] = (chunks, index, patient_name)
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

def process_data(content, file_name):
    logger.info("开始处理电子病历数据")
    logger.info(f"接收到的内容: {content[:200]}...")  # 打印前200个字符

    # 改进的提示词，针对电子病历
    prompt = f"""
    请仔细分析电子病历，并提取所有重要信息。特别注意以下几点：

    1. 首先识别并提取患者的姓名。这是最重要的信息，所有其他信息都应与患者姓名关联。
    2. 提取关键的患者信息，包括但不限于：
       - 年龄
       - 性别
       - 入院日期
       - 出院日期
       - 住院天数
    3. 识别并提取主要诊断信息。
    4. 提取所有相关的医疗信息，包括但不限于：
       - 主诉
       - 现病史
       - 既往史
       - 个人史
       - 家族史
       - 体格检查结果
       - 辅助检查结果（如血常规、影像学检查等）
       - 诊疗经过
       - 用药情况
       - 手术信息（如果有）
    5. 识别任何并发症、特殊情况或注意事项。
    6. 提取出院诊断、出院医嘱等出院相关信息。

    对于每个提取的实体或信息，请建立与患者姓名的直接关系。如果找不到明确的关系类型，请使用"相关"作为默认关系。

    请以JSON格式输出式如下：
    {{
        "patient_name": "患者姓名",
        "entities": [
            {{"name": "实体1", "category": "类别1"}},
            {{"name": "实体2", "category": "类别2"}},
            ...
        ],
        "relations": [
            {{"source": "患者姓名", "relation": "关系", "target": "实体1"}},
            {{"source": "患者姓名", "relation": "关系", "target": "实体2"}},
            ...
        ]
    }}

    请确保每个实体都与患者姓名建立了关系，并尽可能详细地提取信息。对于没有明确关系类型的实体，请使用"属性"作为关系类型。

    电子病历内容：
    {content}
    """

    response = client.chat.completions.create(
        model=get_model_name(),
        messages=[
            {"role": "system", "content": "你是一个专门处理电子病历的AI助手，擅长从复杂的医疗记录中提取关键信息和关系。请尽可能详细地提取所有相关信息，并确保所有信息都与患者姓名建立关系。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        top_p=0.8,
        max_tokens=2048
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

        patient_name = extracted_data['patient_name']
        entities = extracted_data['entities']
        relations = extracted_data['relations']

        # 确保所有实体都是字典，包含name和category
        entities = [e if isinstance(e, dict) else {"name": str(e), "category": "未分类"} for e in entities]
        
        # 确保所有关系都是字典，且包含必要的键
        relations = [
            r if isinstance(r, dict) and all(k in r for k in ['source', 'relation', 'target'])
            else {'source': patient_name, 'relation': "相关", 'target': str(r)}
            for r in relations
        ]

        # 确保所有实体都与患者名字建立了关系
        existing_relations = set((r['source'], r['target']) for r in relations)
        for entity in entities:
            if (patient_name, entity['name']) not in existing_relations and (entity['name'], patient_name) not in existing_relations:
                relations.append({'source': patient_name, 'relation': "相关", 'target': entity['name']})

    except json.JSONDecodeError as e:
        logger.error(f"无法解析OpenAI返回的JSON: {str(e)}")
        # 使用正则表达式提取实体和关
        patient_name = re.search(r'"patient_name":\s*"([^"]+)"', result)
        patient_name = patient_name.group(1) if patient_name else "未知患者"
        entities = [{"name": e, "category": "未分类"} for e in re.findall(r'"name":\s*"([^"]+)"', result)]
        relations = [
            {'source': patient_name, 'relation': "相关", 'target': entity['name']}
            for entity in entities
        ]
    except Exception as e:
        logger.error(f"处理OpenAI返回结果时出错: {str(e)}")
        patient_name = "未知患者"
        entities = []
        relations = []

    logger.info(f"患者姓名: {patient_name}")
    logger.info(f"提取的实体: {entities}")
    logger.info(f"提取的关系: {relations}")

    driver = get_neo4j_driver()
    
    with driver.session() as session:
        # 删除旧数据
        session.run("""
        MATCH (n:Entity)
        WHERE n.source = $file_name
        DETACH DELETE n
        """, file_name=file_name)

        # 创建患者节点
        session.run("""
        MERGE (p:Entity {name: $name, type: 'Patient', source: $file_name})
        SET p.content = $content
        """, name=patient_name, content=content, file_name=file_name)

        # 检查全文索引是否存在
        try:
            index_exists = session.run("""
            CALL db.indexes() YIELD name, labelsOrTypes, properties
            WHERE name = 'entityFulltextIndex'
            RETURN count(*) > 0 AS exists
            """).single()['exists']

            if not index_exists:
                # 尝试创建全文索引
                try:
                    session.run("""
                    CALL db.index.fulltext.createNodeIndex('entityFulltextIndex', ['Entity'], ['name', 'content'])
                    """)
                    logger.info("全文索引创建成功")
                except Exception as e:
                    logger.warning(f"创建全文索引失败: {str(e)}. 继续执行其他操作。")
        except Exception as e:
            logger.warning(f"检查或创建全文索引时出错: {str(e)}. 继续执行其他操作。")

        # 创建其他实体并与患者建立关系
        for entity in entities:
            session.run("""
            MATCH (p:Entity {name: $patient_name, source: $file_name})
            MERGE (e:Entity {name: $entity_name, category: $entity_category, source: $file_name})
            """, patient_name=patient_name, entity_name=entity['name'], entity_category=entity['category'], file_name=file_name)

        for relation in relations:
            session.run("""
            MATCH (s:Entity {name: $source, source: $file_name})
            MATCH (t:Entity {name: $target, source: $file_name})
            MERGE (s)-[r:RELATED_TO {type: $relation_type}]->(t)
            """, source=relation['source'], target=relation['target'], relation_type=relation['relation'], file_name=file_name)

    logger.info(f"处理了 1 个患者节点、{len(entities)} 个实体和 {len(relations)} 个关系")
    driver.close()
    return {"patient_name": patient_name, "entities": entities, "relations": relations}

def query_graph(query):
    logger.info(f"执行图查询: {query}")
    driver = get_neo4j_driver()
    
    with driver.session() as session:
        # 使用标准索引进行查询
        result = session.run("""
        MATCH (n:Entity)
        WHERE n.name CONTAINS $query OR n.content CONTAINS $query
        WITH n
        OPTIONAL MATCH (n)-[r]-(related)
        WHERE related.source IS NOT NULL
        RETURN n.name AS Entity, type(r) AS Relation, r.type AS RelationType, 
               related.name AS Related, n.content AS Content, n.source AS Source,
               related.source AS RelatedSource
        ORDER BY n.name
        LIMIT 20
        """, {"query": query})
        
        entities = set()
        relations = []
        contents = {}
        for record in result:
            entity = record["Entity"]
            related = record["Related"]
            relation_type = record["RelationType"] or record["Relation"]
            
            if entity and related:
                entities.add(entity)
                entities.add(related)
                relation = {
                    "source": entity,
                    "relation": relation_type,
                    "target": related
                }
                if relation not in relations:
                    relations.append(relation)
                
                # 添加反向关系
                reverse_relation = {
                    "source": related,
                    "relation": f"反向_{relation_type}",
                    "target": entity
                }
                if reverse_relation not in relations:
                    relations.append(reverse_relation)
            
            if entity:
                contents[entity] = record["Content"]
            if related:
                contents[related] = record["Content"]
    
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
            return "抱歉，我没有找到与您的问题相关的信息。请尝试用不同的方式提问，或者确认所查询的信息是否已经录入系。", [], []
        
        # 处理间接关系
        related_entities = set()
        for relation in relations:
            related_entities.add(relation['source'])
            related_entities.add(relation['target'])
        
        context = f"基于以下实体信息：\n"
        for entity in related_entities:
            content = contents.get(entity, '无详细信息')
            if content is not None:
                context += f"{entity}: {content[:200]}...\n"
            else:
                context += f"{entity}: 无详细信息\n"
        context += "相关关系：\n"
        for relation in relations:
            context += f"{relation['source']} {relation['relation']} {relation['target']}\n"
        
        prompt = f"{context}\n\n请根据上述信息回答问题：{query}\n\n回答："
        
        logger.info(f"发送到 OpenAI 的提示: {prompt}")
        
        response = client.chat.completions.create(
            model=get_model_name(),
            messages=[
                {"role": "system", "content": "你是一个医疗助手，根据给定的实体信息和关系准确回答问题。请直接使用提供的信息，不要添加未给出的假设。如果信息不足，请如说明。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            top_p=0.8,
            max_tokens=150
        )
        
        logger.info(f"OpenAI 响应: {response}")
        
        if response and response.choices and len(response.choices) > 0 and response.choices[0].message:
            answer = response.choices[0].message.content.strip()
            if not answer:
                answer = "抱歉，我无法根据提供的信息回答这个问题。请尝试提供更多细节或以不同的方式提问。"
        else:
            answer = "抱歉，处理您的问题时出现了意外情况。请稍后再试。"
        
        logger.info(f"搜索结果: {answer}")
        return answer, list(entities), relations
    except Exception as e:
        logger.error(f"混合搜索过程中发生误: {str(e)}", exc_info=True)
        return f"抱歉，在处理您的问题时发生了错误: {str(e)}", [], []

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
    logger.info(f"查询实体 {entity_name} 的关信息")
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

def generate_final_answer(query, graph_answer, vector_answer, fulltext_results, excerpt, graph_entities, graph_relations):
    prompt = f"""
    基于以下信息，请回答问题并提供详细的推理过程：

    问题：{query}

    图数据库回答：{graph_answer}
    图数据库实体：{', '.join(graph_entities)}
    图数据库关系：{', '.join([f"{r['source']} --[{r['relation']}]--> {r['target']}" for r in graph_relations])}

    向量数据库回答：{vector_answer}

    全文检索结果：
    {' '.join([f"文档: {result['title']}, 相关度: {result['score']:.2f}, 匹配内容: {result['highlights']}" for result in fulltext_results[:5]])}

    全文检索匹配文档数量：{len(fulltext_results)}

    相关原文：{excerpt}

    请提供一个综合的回答，包括：
    1. 直接回答问题，综合考虑所有数据源的信息，特别是图数据库中的关系信息和全文检索的结果
    2. 明确指出所有相关患者的姓名，即使他们的情况可能略有不同
    3. 对每个相关患者的情况进行简要说明
    4. 详细的推理过程，解释如何得出结论，包括对可能存在的不确定性或不一致的讨论
    5. 使用的证据及其来源（包括图数据库、向量数据库和全文检索）

    注意：
    - 请确保考虑所有检索到的相关信息，即使某些信息可能看起来不太直接相关
    - 如果发现任何潜在的相关信息，即使不确定，也请在回答中提及并说明原因
    - 图数据库的信息通常更精确，请优先考虑图数据库中的关系信息
    - 全文检索结果提供了直接的文本匹配，请充分利用这些信息，特别是在识别症状和患者时
    - 向量数据库的结果可能提供额外的相关信息
    - 请确保在回答中明确提到所有可能相关的患者姓名，并直接引用全文检索的结果

    请确保回答全面且准确，不要忽视任何重要信息，特别是全文检索中直接匹配的内容。如果信息不足或存在不确定性，请在回答中明确指出。
    """
    
    response = client.chat.completions.create(
        model=get_model_name(),
        messages=[
            {"role": "system", "content": "你是个智能助手，能够综合分析来自不同数据源的信息，并提供准确、全面的回答。你需要仔细考虑所有提供的信息，特别是要注意全文检索的直接匹配结果和图数据库中的关系信息。即使某些信息可能看起来不太直接相关，也请在回答中提及并解释其潜在相关性。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        top_p=0.8,
        max_tokens=1000  # 增加 token 限制以获取更详细的回答
    )
    
    return response.choices[0].message.content.strip()

def vector_search(query, k=5):
    global faiss_index, faiss_id_to_text
    if faiss_index is None:
        logger.error("FAISS 索引未初始化")
        return []
    
    query_vector = model.encode([query])
    D, I = faiss_index.search(query_vector.astype('float32'), k)
    
    results = []
    for i, d in zip(I[0], D[0]):
        if i != -1 and i in faiss_id_to_text:
            text = faiss_id_to_text[i]
            patient_name = text.split('。')[0].split('：')[1]  # 从文本块中提取患者姓名
            results.append({
                "text": text,
                "distance": float(d),
                "patient_name": patient_name
            })
    
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
    'faiss_id_counter',
    'QueryParser',
    'delete_graph_data',
    'delete_vector_data',
    'delete_fulltext_index',
]

def fused_graph_vector_search(query, graph_db, vector_db):
    # 1. 从查询中提取实体
    entities = extract_entities(query)
    
    # 2. 使用实体在图数据库中进行子图匹配
    subgraph = graph_db.match_subgraph(entities)
    
    # 3. 将子图信息换为文本
    subgraph_text = convert_subgraph_to_text(subgraph)
    
    # 4. 在向量数据库中搜索相关文档
    relevant_docs = vector_db.search(query + " " + subgraph_text)
    
    # 5. 结合子图和相关文档生成最终答案
    final_answer = generate_answer(query, subgraph, relevant_docs)
    
    return final_answer

class CustomChineseAnalyzer(Analyzer):
    def __init__(self):
        self.analyzer = JiebaAnalyzer()

    def __call__(self, text, **kwargs):
        tokens = self.analyzer(text)
        for t in tokens:
            yield Token(text=t.text, pos=t.pos, startchar=t.startchar, endchar=t.endchar)

def openai_tokenize(text):
    response = client.chat.completions.create(
        model=get_model_name(),
        messages=[
            {"role": "system", "content": "你是一个专门用于中文分词的AI助手。请对给定的文本进行分词，特别注意医学术语。"},
            {"role": "user", "content": f"请对以下文本进行分词，返回一个JSON格式的词语列表。文本：{text[:1000]}"}  # 限制文本长度以避免超过token限制
        ],
        max_tokens=1000
    )
    tokens = json.loads(response.choices[0].message.content)
    return tokens

class OpenAIAnalyzer(Analyzer):
    def __call__(self, text, **kwargs):
        tokens = openai_tokenize(text)
        for t in tokens:
            yield Token(text=t, pos=len(t))

def create_fulltext_index(content, file_name):
    logger.info(f"开始为文件 {file_name} 创建或更新全文索")
    logger.info(f"文件内容长: {len(content)}")
    logger.info(f"文件内容前200字符: {content[:200]}")
    
    if not os.path.exists("fulltext_index"):
        os.mkdir("fulltext_index")
        logger.info("创建了新的全文索引目录")
    
    schema = Schema(title=TEXT(stored=True), content=TEXT(stored=True, analyzer=OpenAIAnalyzer()))
    
    if not os.path.exists("fulltext_index/MAIN_WRITELOCK"):
        ix = create_in("fulltext_index", schema)
    else:
        ix = open_dir("fulltext_index")
    
    writer = ix.writer()
    writer.add_document(title=file_name, content=content)
    writer.commit()
    
    logger.info(f"全文索引更新完成，文件名: {file_name}")
    
    # 验证索引内容
    with ix.searcher() as searcher:
        results = searcher.search(Term("title", file_name))
        if results:
            logger.info(f"成功检索到文档: {file_name}")
            logger.info(f"索引中的文档内容长度: {len(results[0]['content'])}")
            logger.info(f"索引中的文档内容前200字符: {results[0]['content'][:200]}")
        else:
            logger.warning(f"无法检索到文档: {file_name}")
    
    return ix

def search_fulltext_index(query):
    logger.info(f"开始全文检索，查询: {query}")
    if not os.path.exists("fulltext_index"):
        logger.warning("全文索引目录不存在，无法执行搜索")
        return []

    ix = open_dir("fulltext_index")
    with ix.searcher() as searcher:
        keywords = extract_core_keywords(query)
        logger.info(f"提取的核心关键词: {keywords}")

        query_parser = QueryParser("content", ix.schema, group=OrGroup.factory(0.9))
        query_terms = []
        for keyword in keywords:
            query_terms.append(Term("content", keyword))
        
        final_query = query_parser.parse(" OR ".join(keywords))
        logger.info(f"构建的查询: {final_query}")

        results = searcher.search(final_query, limit=None)
        logger.info(f"搜索结果数量: {len(results)}")
        
        return [{"title": r["title"], 
                 "score": r.score, 
                 "highlights": r.highlights("content", top=5),  # 增加返回的高亮片段数量
                 "content": r.get("content", "")[:500]  # 增加返回的内容长度
                } for r in results]

def extract_core_keywords(query):
    response = client.chat.completions.create(
        model=get_model_name(),
        messages=[
            {"role": "system", "content": "你是一个专门用于提取医疗领域核心关键词的AI助手。请从给定的问题中提取最重要的医学术语或症状描述。"},
            {"role": "user", "content": f"""
            从以下问题中提取2-3个最重要的核心关键词。这些关键词应该是搜索医疗文档时最有可能找到相关信息的词。
            
            注意：
            1. 常见词"患者"、"病人"、"医生"、"医院"等不应被视为核心关键词，除非它们是问题的主要焦点。
            2. 优先选择专业医学术语、症状描述或特定的疾病名称。
            3. 关键词可以是题中明确出现的词，也可以是根据问题内容推断出的相关医学术语。
            4. 如果问题中没有明确的医学术语，可以选择问题中最具体、最相关的词语。

            问题：{query}

            核心关键词：
            """}
        ],
        max_tokens=50
    )
    keywords = response.choices[0].message.content.strip().split(', ')
    return [keyword.strip() for keyword in keywords if keyword.strip()]  # 移除空字符串

medical_synonyms = {
    "脑梗死": ["脑梗塞", "缺血性脑卒中"],
    "白细胞": ["白血球", "WBC"],
    # 添更多同义词...
}

def expand_query(query):
    expanded_terms = [query]
    for term in query.split():
        if term in medical_synonyms:
            expanded_terms.extend(medical_synonyms[term])
    return " OR ".join(expanded_terms)

def check_index_content(term):
    ix = open_dir("fulltext_index")
    with ix.searcher() as searcher:
        results = searcher.search(Term("content", term))
        logger.info(f"搜索 '{term}' 的结果数量: {len(results)}")
        for hit in results:
            logger.info(f"文档: {hit['title']}")
            logger.info(f"内容片段: {hit.highlights('content')}")

def delete_graph_data(file_name):
    driver = get_neo4j_driver()
    with driver.session() as session:
        # 删除与文件相关的所有节点和关系
        session.run("""
        MATCH (n:Entity)
        WHERE n.source = $file_name
        DETACH DELETE n
        """, file_name=file_name)
    logger.info(f"已删除图数据库中与文件 {file_name} 相关的数据")

def delete_vector_data(file_name):
    global faiss_index, faiss_id_to_text, faiss_id_counter
    if file_name in st.session_state.file_indices:
        chunks, _, _ = st.session_state.file_indices[file_name]  # 解包三个元素，忽略 index 和 patient_name
        # 从 faiss_id_to_text 中删除相关条目
        faiss_id_to_text = {k: v for k, v in faiss_id_to_text.items() if v not in chunks}
        # 重建 FAISS 索引
        new_index = faiss.IndexFlatL2(384)  # 384 是向量维度，根据实际情况调整
        new_vectors = []
        new_id_to_text = {}
        new_id = 0
        for id, text in faiss_id_to_text.items():
            try:
                vector = faiss_index.reconstruct(id)
                new_vectors.append(vector)
                new_id_to_text[new_id] = text
                new_id += 1
            except RuntimeError:
                logger.warning(f"无法重构ID为{id}的向量，已跳过")
        
        if new_vectors:
            new_index.add(np.array(new_vectors))
            faiss_index = new_index
            faiss_id_to_text = new_id_to_text
            faiss_id_counter = new_id
        else:
            logger.warning("删除后没有剩余向量，FAISS索引已清空")
            faiss_index = faiss.IndexFlatL2(384)
            faiss_id_to_text = {}
            faiss_id_counter = 0
        
        # 从 session_state 中删除文件索引
        del st.session_state.file_indices[file_name]
    else:
        logger.warning(f"文件 {file_name} 不在索引中")
    
    logger.info(f"已删除向量数据库中与文件 {file_name} 相关的数据")

def delete_fulltext_index(file_name):
    if os.path.exists("fulltext_index"):
        ix = open_dir("fulltext_index")
        writer = ix.writer()
        
        # 删除与文件名完全匹配的文档
        writer.delete_by_term('title', file_name)
        
        # 删除所有以文件名开头的文档（处理分块的情况）
        for doc_num in writer.reader().all_doc_ids():
            doc = writer.reader().stored_fields(doc_num)
            if doc['title'].startswith(file_name):
                writer.delete_document(doc_num)
        
        writer.commit()
        
        logger.info(f"已删除全文索引中与文件 {file_name} 相关的所有数据")
    else:
        logger.warning("全文索引目录不存在")
    
    # 验证删除操作
    if os.path.exists("fulltext_index"):
        ix = open_dir("fulltext_index")
        with ix.searcher() as searcher:
            remaining_docs = [doc for doc in searcher.all_stored_fields() if doc['title'].startswith(file_name)]
            if remaining_docs:
                logger.warning(f"删除操作后仍有与 {file_name} 相关的文档: {[doc['title'] for doc in remaining_docs]}")
            else:
                logger.info(f"成功删除所有与 {file_name} 相关的文档")

def clear_vector_data():
    global faiss_index, faiss_id_to_text, faiss_id_counter
    faiss_index = initialize_faiss()
    faiss_id_to_text = {}
    faiss_id_counter = 0
    
    # 删除所有保存的索引文件
    if os.path.exists('indices'):
        for file in os.listdir('indices'):
            os.remove(os.path.join('indices', file))
    
    logger.info("所有向量数据已被清除")

def initialize_neo4j():
    try:
        # 确保配置已设置
        if not CURRENT_NEO4J_CONFIG or not all(CURRENT_NEO4J_CONFIG.values()):
            logger.error("Neo4j 配置未正确设置")
            # 使用默认配置
            CURRENT_NEO4J_CONFIG.update({
                "URI": AURA_URI,
                "USERNAME": AURA_USERNAME,
                "PASSWORD": AURA_PASSWORD
            })
            logger.info("已设置默认 Neo4j 配置")

        driver = get_neo4j_driver()
        with driver.session() as session:
            # 测试连接
            try:
                session.run("RETURN 1 AS test").single()
                logger.info("Neo4j 连接测试成功")
            except Exception as e:
                logger.error(f"Neo4j 连接测试失败: {str(e)}")
                raise

            # 检查并创建约束和索引，使用单独的事务
            try:
                # 创建唯一约束
                session.run("""
                DROP CONSTRAINT entity_name_unique IF EXISTS
                """)
                session.run("""
                CREATE CONSTRAINT entity_name_unique IF NOT EXISTS
                FOR (n:Entity)
                REQUIRE n.name IS UNIQUE
                """)
                logger.info("实体名称唯一约束创建成功")

                # 创建索引（使用单独的事务）
                session.run("""
                DROP INDEX entity_name_index IF EXISTS
                """)
                session.run("""
                CREATE INDEX entity_name_index IF NOT EXISTS
                FOR (n:Entity)
                ON (n.name)
                """)
                logger.info("实体名称索引创建成功")

                # 创建内容索引（使用单独的事务）
                session.run("""
                DROP INDEX entity_content_index IF EXISTS
                """)
                session.run("""
                CREATE INDEX entity_content_index IF NOT EXISTS
                FOR (n:Entity)
                ON (n.content)
                """)
                logger.info("实体内容索引创建成功")

            except Exception as e:
                logger.error(f"创建索引和约束时出错: {str(e)}")
                # 尝试清理可能的部分创建
                try:
                    session.run("DROP CONSTRAINT entity_name_unique IF EXISTS")
                    session.run("DROP INDEX entity_name_index IF EXISTS")
                    session.run("DROP INDEX entity_content_index IF EXISTS")
                except:
                    pass
                raise

        driver.close()
        logger.info("Neo4j 初始化完成")
    except Exception as e:
        logger.error(f"初始化 Neo4j 时出错: {str(e)}")
        raise