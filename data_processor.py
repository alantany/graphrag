import jieba
import jieba.posseg as pseg
import re
from neo4j import GraphDatabase
import logging
import openai
import json
from openai import OpenAI

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Neo4j 配置
NEO4J_CONFIG = {
    "LOCAL_URI": "bolt://localhost:7687",
    "LOCAL_USERNAME": "test",
    "LOCAL_PASSWORD": "Mikeno01",
    "AURA_URI": "neo4j+s://b76a61f2.databases.neo4j.io:7687",
    "AURA_USERNAME": "neo4j",
    "AURA_PASSWORD": "JkVujA4SZWdifvfvj5m_gwdUgHsuTxQjbJQooUl1C14"
}

# 在文件顶部附近定义
CURRENT_NEO4J_CONFIG = {}

client = None

def set_neo4j_config(uri_key, username_key, password_key):
    global CURRENT_NEO4J_CONFIG
    logger.info(f"正在设置 Neo4j 配置，使用键: URI={uri_key}, USERNAME={username_key}, PASSWORD={password_key}")
    try:
        CURRENT_NEO4J_CONFIG = {
            "URI": NEO4J_CONFIG[uri_key],
            "USERNAME": NEO4J_CONFIG[username_key],
            "PASSWORD": NEO4J_CONFIG[password_key]
        }
        logger.info(f"Neo4j 配置已设置: {CURRENT_NEO4J_CONFIG}")
    except KeyError as e:
        logger.error(f"设置 Neo4j 配置时出错: 找不到键 {str(e)}")
        logger.error(f"当前 NEO4J_CONFIG: {NEO4J_CONFIG}")
        CURRENT_NEO4J_CONFIG = {}
    
    logger.info(f"设置后的 CURRENT_NEO4J_CONFIG: {CURRENT_NEO4J_CONFIG}")

def initialize_openai(api_key, base_url):
    global client
    client = OpenAI(api_key=api_key, base_url=base_url)
    logger.info("OpenAI 初始化完成")

def process_data(content):
    logger.info("开始处理数据")
    logger.info(f"接收到的内容: {content[:200]}...")  # 打印前200个字符

    # 改进的提示词
    prompt = f"""
    请从以下医疗记录中提取所有重要的实体和关系。
    实体应包括但不限于：患者姓名、年龄、性别、诊断、症状、检查、治疗、药物、生理指标等。
    关系应描述实体之间的所有可能联系，如"患有"、"接受检查"、"使用药物"、"属性"等。
    请确保每个实体都至少有一个关系。对于没有明确关系的属性（如性别、年龄等），请使用"属性"作为关系类型。
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

        # 确保所有实体都是字符串
        entities = [str(e) for e in entities]
        
        # 确保所有关系都是字典，且包含必要的键
        relations = [
            r if isinstance(r, dict) and all(k in r for k in ['source', 'relation', 'target'])
            else {'source': str(r[0]), 'relation': str(r[1]), 'target': str(r[2])}
            for r in relations
        ]

    except json.JSONDecodeError as e:
        logger.error(f"无法解析OpenAI返回的JSON: {str(e)}")
        # 使用正则表达式提取实体和关系
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

    driver = GraphDatabase.driver(
        CURRENT_NEO4J_CONFIG["URI"],
        auth=(CURRENT_NEO4J_CONFIG["USERNAME"], CURRENT_NEO4J_CONFIG["PASSWORD"])
    )
    
    with driver.session() as session:
        # 创建实体
        for entity in entities:
            result = session.run("MERGE (n:Entity {name: $name}) RETURN count(*)", name=str(entity).strip())
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
            RETURN count(*)
            """, source=str(source).strip(), target=str(target).strip(), rel_type=str(rel_type).strip())
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
    
    # 分解查询字符串为关键词
    keywords = jieba.lcut(query)
    logger.info(f"查询关键词: {keywords}")
    
    with driver.session() as session:
        result = session.run("""
        MATCH (n:Entity)
        WHERE any(keyword IN $keywords WHERE toLower(n.name) CONTAINS toLower(keyword))
        OPTIONAL MATCH (n)-[r]-(m)
        RETURN n.name AS Entity, type(r) AS Relation, r.type AS RelationType, m.name AS Related
        """, {"keywords": keywords})
        
        entities = set()
        relations = []
        for record in result:
            if record["Entity"]:
                entities.add(record["Entity"])
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
    return list(entities), relations

def hybrid_search(query):
    logger.info(f"执行混合搜索: {query}")
    try:
        entities, relations = query_graph(query)
        
        if not entities and not relations:
            return "抱歉，我没有找到与您的问题相关的信息。请尝试用不同的方式提问，或者确认所查询的信息是否已经录入系统。"
        
        # 限制上下文大小
        max_entities = 10
        max_relations = 20
        context = f"基于以下实体信息：{', '.join(entities[:max_entities])}\n"
        context += "相关关系：\n"
        for relation in relations[:max_relations]:
            context += f"{relation['source']} {relation['relation']} {relation['target']}\n"
        
        prompt = f"{context}\n\n请根据上述信息回答问题：{query}\n\n回答："
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个医疗助手，根据给定的实体信息和关系准确回答问题。如果信息不足，请如实说明。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        
        answer = response.choices[0].message.content.strip()
        if not answer:
            answer = "抱歉，我无法根据提供的信息回答这个问题。请尝试提供更多细节或以不同的方式提问。"
        logger.info(f"搜索结果: {answer}")
        return answer
    except Exception as e:
        logger.error(f"混合搜索过程中发生错误: {str(e)}")
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

# 更新 __all__ 列表
__all__ = ['set_neo4j_config', 'initialize_openai', 'process_data', 'query_graph', 'hybrid_search', 'CURRENT_NEO4J_CONFIG', 'query_graph_with_entities', 'get_entity_relations']