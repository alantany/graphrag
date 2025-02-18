from neo4j import GraphDatabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_neo4j_driver():
    return GraphDatabase.driver(
        "neo4j+s://85c689ad.databases.neo4j.io:7687",
        auth=("neo4j", "XL37Q-0UhF1YA2diY3f9Ah3dLxHmWlyoN6rexDu9sdA")
    )

def check_database():
    driver = get_neo4j_driver()
    
    with driver.session() as session:
        # 检查所有实体
        result = session.run("""
        MATCH (n:Entity)
        RETURN n.name as name, n.category as category, n.source as source
        LIMIT 10
        """)
        
        print("\n=== 数据库中的实体 ===")
        entities = list(result)
        if not entities:
            print("数据库中没有找到任何实体！")
        else:
            for record in entities:
                print(f"名称: {record['name']}, 类别: {record['category']}, 来源: {record['source']}")
        
        # 检查所有关系
        result = session.run("""
        MATCH (n:Entity)-[r:RELATED_TO]->(m:Entity)
        RETURN n.name as source, r.type as relation, m.name as target
        LIMIT 10
        """)
        
        print("\n=== 实体之间的关系 ===")
        relations = list(result)
        if not relations:
            print("数据库中没有找到任何关系！")
        else:
            for record in relations:
                print(f"{record['source']} --[{record['relation']}]--> {record['target']}")
    
    driver.close()

if __name__ == "__main__":
    check_database() 