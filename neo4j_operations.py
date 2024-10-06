from neo4j import GraphDatabase
import logging

class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]

    def create_node(self, label, properties):
        query = (
            f"CREATE (n:{label} $properties) "
            "RETURN n"
        )
        return self.query(query, parameters={'properties': properties})

    def create_relationship(self, start_node_label, start_node_property, end_node_label, end_node_property, relationship_type, properties=None):
        query = (
            f"MATCH (a:{start_node_label}), (b:{end_node_label}) "
            f"WHERE a.{start_node_property} = $start_value AND b.{end_node_property} = $end_value "
            f"CREATE (a)-[r:{relationship_type} $properties]->(b) "
            "RETURN type(r)"
        )
        return self.query(query, parameters={
            'start_value': start_node_property,
            'end_value': end_node_property,
            'properties': properties or {}
        })

    def get_node(self, label, property_name, property_value):
        query = (
            f"MATCH (n:{label}) "
            f"WHERE n.{property_name} = $value "
            "RETURN n"
        )
        return self.query(query, parameters={'value': property_value})

# Neo4j连接配置
AURA_CONFIG = {
    "uri": "neo4j+s://b76a61f2.databases.neo4j.io:7687",
    "username": "neo4j",
    "password": "JkVujA4SZWdifvfvj5m_gwdUgHsuTxQjbJQooUl1C14"
}

LOCAL_CONFIG = {
    "uri": "bolt://localhost:7687",
    "username": "test",
    "password": "Mikeno01"
}

def get_neo4j_connection(config):
    return Neo4jConnection(config["uri"], config["username"], config["password"])

# 使用示例
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # 这里使用本地配置作为示例
        conn = get_neo4j_connection(LOCAL_CONFIG)
        logger.info("Successfully connected to Neo4j database")

        # 测试创建节点
        result = conn.create_node("Person", {"name": "John Doe", "age": 30})
        logger.info(f"Created node: {result}")

        # 测试查询节点
        result = conn.get_node("Person", "name", "John Doe")
        logger.info(f"Retrieved node: {result}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

    finally:
        if conn:
            conn.close()
            logger.info("Neo4j connection closed")