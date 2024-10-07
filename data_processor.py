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

# 添加 FAISS 相关的全局变量
faiss_index = None
faiss_id_to_text = {}
faiss_id_counter = 0

def set_neo4j_config(config_type):
    global CURRENT_NEO4J_CONFIG
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
    
    logger.info(f"Neo4j 配置已设置: URI={CURRENT_NEO4J_CONFIG['URI']}, USERNAME={CURRENT_NEO4J_CONFIG['USERNAME']}")
    return True

def initialize_openai(api_key, base_url):
    global client
    client = OpenAI(api_key=api_key, base_url=base_url)
    logger.info("OpenAI 初始化完成")

def initialize_faiss():
    global faiss_index
    faiss_index = faiss.IndexFlatL2(1536)  # OpenAI's embedding dimension
    faiss_index = faiss.IndexIDMap(faiss_index)
    logger.info("FAISS 初始化完成")

# 向量化文本
def get_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-ada-002")
    return response.data[0].embedding

def process_data(content):
    logger.info("开始处理数据")
    logger.info(f"接收到的内容: {content[:200]}...")  # 打印前200个字符

    # 改进的提示词
    prompt = f"""
    请从以下医疗记录中提取所有要的实体和关系。
    实体应包括但不限于：患者姓名、年龄、性别、诊断、症状、检查、治疗、药物、生理指标等。
    关系应描述实体之间的所有可能联系，如"患有"、"接受检查"、"使用药物"、"属性"等。
    请确保每个实体都至少有一个关系。对于没有明确关系的性（如性别、年龄等），请使用"属性"作为关系类型。
    请尽可能详细地提取关系，不要遗漏任何可能的连接。
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
            {"role": "system", "content": "你是一个医疗信息提取助手，擅长从医疗记录中提取实体和关系。请尽可能详细地提取所有相关信息。"},
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
        # 使用则表达式提取实体和关系
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
                elif entity in ["血糖", "血压", "体重", "心率", "体温"]:
                    relations.append({"source": patient_name, "relation": "生理指标", "target": entity})
                elif entity in ["口干", "多尿", "多食", "体重下降"]:
                    relations.append({"source": patient_name, "relation": "症状", "target": entity})
                elif "检查" in entity or entity in ["心电图", "胸片", "肌电图", "超声", "眼科检查", "GFR", "CGMS"]:
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
            logger.info(f"创建或更新实体: {entity}, 影响的节点数: {count}")
        
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
            return "抱歉，我没有找到与您的问题相关的信息。请尝试用不同的方式提问，或者确认所查询的信息是否已经录入系统。"
        
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
                answer = "抱歉，我无法根��提供的信息回答这个问题。请尝试提供更多细节或以不同的方式提问。"
        else:
            answer = "抱歉，处理您的问题时出现了意外况。请稍后再试。"
        
        logger.info(f"搜索结果: {answer}")
        return answer
    except Exception as e:
        logger.error(f"混合搜索过程中发生错误: {str(e)}", exc_info=True)
        return f"抱歉，在处理您的问题时发生了错误: {str(e)}"

# 在文件末尾添加以下函数

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
            logger.info(f"实体 {entity_name} 的查询结果: {query_result}")
            return query_result
    except Exception as e:
        logger.error(f"查询实体 {entity_name} 时发生错误: {str(e)}")
        return []
    finally:
        driver.close()

# 处理数据并存入 FAISS
def process_data_vector(content):
    global faiss_id_counter
    logger.info("开始处理数据（向量化）")
    logger.info(f"接收到的内容: {content[:200]}...")  # 打印前200个字符

    # 使用 OpenAI 提取实体和关系
    prompt = f"""
    请从以下医疗记录中提取重要的实体和关系。
    实体应包括但不限于：患者姓名、年龄、性别、诊断、症状、检查、治疗、药物等。
    关系应描述实体之间的联系，如"患有"、"接受检查"、"使用药物"等。
    请以JSON格式输出，格式如下：
    {{
        "entities": ["实体1", "实体2", ...],
        "relations": [
            {{"source": "实体1", "relation": "关系", "target": "实体2"}},
            ...
        ]
    }}

    医疗记录：
    {content}
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "你是一个医疗信息提取助手，擅长从医疗记录中提取实体和关系。"},
            {"role": "user", "content": prompt}
        ]
    )

    response_content = response.choices[0].message.content
    logger.info(f"OpenAI API 返回的原始内容: {response_content}")

    try:
        # ��试解析 JSON
        result = json.loads(response_content)
    except json.JSONDecodeError:
        # 如果解析失败，尝试使用正则表达式提取实体和关系
        logger.warning("JSON 解析失败，尝试使用正则表达式提取信息")
        entities = re.findall(r'"([^"]+)"', response_content)
        relations = re.findall(r'\{"source": "([^"]+)", "relation": "([^"]+)", "target": "([^"]+)"\}', response_content)
        result = {
            "entities": entities,
            "relations": [{"source": s, "relation": r, "target": t} for s, r, t in relations]
        }

    entities = result['entities']
    relations = result['relations']

    # 批量获取实体和关系的向量
    all_texts = entities + [f"{r['source']} {r['relation']} {r['target']}" for r in relations]
    
    # 批量获取嵌入
    embeddings = get_embeddings(all_texts)

    # 为每个实体和关系创建向量
    for i, embedding in enumerate(embeddings):
        faiss_index.add_with_ids(np.array([embedding]).astype('float32'), np.array([faiss_id_counter]))
        faiss_id_to_text[faiss_id_counter] = all_texts[i]
        faiss_id_counter += 1

    logger.info(f"处理了 {len(entities)} 个实体和 {len(relations)} 个关系")
    return {"entities": entities, "relations": relations}

def get_embeddings(texts):
    # 批量获取嵌入
    response = client.embeddings.create(input=texts, model="text-embedding-ada-002")
    return [item.embedding for item in response.data]

def vector_search(query, k=5):
    logger.info(f"执行向量搜索: {query}")
    query_vector = get_embedding(query)
    D, I = faiss_index.search(np.array([query_vector]).astype('float32'), k)
    results = [faiss_id_to_text[i] for i in I[0] if i in faiss_id_to_text]
    logger.info(f"向量搜索结果: {results}")
    return results

def hybrid_search_with_vector(query):
    logger.info(f"执行混合搜索（包含向量）: {query}")
    try:
        graph_entities, graph_relations, graph_contents = query_graph(query)
        vector_results = vector_search(query)
        
        context = f"基于图数据库的信息：\n"
        for entity in graph_entities[:5]:
            content = graph_contents.get(entity, '无详细信息')
            if content is not None:
                context += f"{entity}: {content[:200]}...\n"
            else:
                context += f"{entity}: 无详细信息\n"
        context += "相关关系：\n"
        for relation in graph_relations[:5]:
            context += f"{relation['source']} {relation['relation']} {relation['target']}\n"
        
        context += "\n基于向量数据库的相关信息：\n"
        for result in vector_results:
            context += f"- {result}\n"
        
        prompt = f"{context}\n\n请根据上���信息回答问题：{query}\n\n回答："
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个医疗助手，根据给定的图数据库信息和向量数据库信息准确回答问题。如果信息不足，请如实说明。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        
        answer = response.choices[0].message.content.strip()
        logger.info(f"混合搜索结果: {answer}")
        return answer
    except Exception as e:
        logger.error(f"混合搜索过程中发生错误: {str(e)}", exc_info=True)
        return f"抱歉，在处理您的问题时发生了错误: {str(e)}"

def faiss_query(query, k=5):
    logger.info(f"执行 FAISS 查询: {query}")
    query_vector = get_embedding(query)
    D, I = faiss_index.search(np.array([query_vector]).astype('float32'), k)
    results = [
        {"id": int(i), "text": faiss_id_to_text.get(int(i), "Unknown"), "distance": float(d)}
        for i, d in zip(I[0], D[0]) if int(i) in faiss_id_to_text
    ]
    logger.info(f"FAISS 查询结果: {results}")
    return results

def get_all_faiss_documents():
    logger.info("获取所��� FAISS 文档")
    all_documents = []
    for id, text in faiss_id_to_text.items():
        all_documents.append({"id": id, "text": text})
    logger.info(f"FAISS 中有 {len(all_documents)} 个文档")
    return all_documents

# 在文件的适当位置添加这个函数，比如在 set_neo4j_config 函数之后
def get_neo4j_driver():
    return GraphDatabase.driver(
        CURRENT_NEO4J_CONFIG["URI"],
        auth=(CURRENT_NEO4J_CONFIG["USERNAME"], CURRENT_NEO4J_CONFIG["PASSWORD"])
    )

# 确保将这个函数添加到 __all__ 列表中
__all__.append('get_neo4j_driver')
# 更新 __all__ 列表
__all__ = ['set_neo4j_config', 'initialize_openai', 'process_data', 'query_graph', 'hybrid_search', 'CURRENT_NEO4J_CONFIG', 'query_graph_with_entities', 'get_entity_relations', 'initialize_faiss', 'process_data_vector', 'vector_search', 'hybrid_search_with_vector', 'faiss_query', 'get_all_faiss_documents']