from typing import List, Dict, Any
from neo4j import GraphDatabase
import logging
import json

class GraphBuilder:
    def __init__(self, neo4j_connection):
        self.neo4j_connection = neo4j_connection
        self.logger = logging.getLogger(__name__)

    def clear_existing_data(self):
        """
        安全地删除之前导入的数据，只删除TextChunk节点和相关关系
        """
        clear_query = """
        MATCH (n:TextChunk)
        DETACH DELETE n
        """
        self.neo4j_connection.query(clear_query)

    def build_graph(self, processed_data: Dict[str, List[Dict[str, Any]]]):
        """
        将处理后的数据构建成知识图谱
        """
        self.clear_existing_data()
        self.logger.info(f"Creating {len(processed_data['nodes'])} nodes")
        self._create_nodes(processed_data['nodes'])
        self.logger.info(f"Creating {len(processed_data['relationships'])} relationships")
        self._create_relationships(processed_data['relationships'])
        self.logger.info("Graph building completed")

    def _create_nodes(self, nodes: List[Dict[str, Any]]):
        """
        创建节点
        """
        for node in nodes:
            # 确保 embedding 是 JSON 可序列化的
            node_properties = {
                "chunk_id": node["chunk_id"],
                "file_name": node["file_name"],
                "content": node["content"],
                "embedding": json.dumps(node["embedding"])  # 将 embedding 转换为 JSON 字符串
            }
            result = self.neo4j_connection.create_node("TextChunk", node_properties)
            self.logger.info(f"Created node: {result}")

    def _create_relationships(self, relationships: List[Dict[str, Any]]):
        """
        创建关系
        """
        for rel in relationships:
            result = self.neo4j_connection.create_relationship(
                "TextChunk", "chunk_id",
                "TextChunk", "chunk_id",
                rel["type"],
                {"file_name": rel["properties"]["file_name"]}
            )
            self.logger.info(f"Created relationship: {result}")

    def add_entity_nodes(self, entities: List[Dict[str, Any]]):
        """
        添加实体节点并创建与文本块的关系
        """
        for entity in entities:
            self.neo4j_connection.create_node("Entity", {
                "name": entity["name"],
                "type": entity["type"]
            })
            self.neo4j_connection.create_relationship(
                "TextChunk", "chunk_id",
                "Entity", "name",
                "CONTAINS",
                {"confidence": entity["confidence"]}
            )

    def add_custom_relationships(self, custom_relationships: List[Dict[str, Any]]):
        """
        添加自定义关系
        """
        for rel in custom_relationships:
            self.neo4j_connection.create_relationship(
                rel["start_node_type"], rel["start_node_property"],
                rel["end_node_type"], rel["end_node_property"],
                rel["relationship_type"],
                rel.get("properties", {})
            )

# 使用示例
if __name__ == "__main__":
    from neo4j_operations import get_neo4j_connection, LOCAL_CONFIG
    from data_processor import DataProcessor

    # 模拟数据
    file_indices = {
        "example.txt": (["This is a test document."], faiss.IndexFlatL2(384))
    }

    # 处理数据
    processor = DataProcessor()
    processed_data = processor.process_file_indices(file_indices)

    # 连接到Neo4j
    neo4j_conn = get_neo4j_connection(LOCAL_CONFIG)

    # 构建图
    graph_builder = GraphBuilder(neo4j_conn)
    graph_builder.build_graph(processed_data)

    # 添加一些示例实体
    entities = [
        {"name": "test", "type": "keyword", "confidence": 0.9}
    ]
    graph_builder.add_entity_nodes(entities)

    # 添加一些自定义关系
    custom_relationships = [
        {
            "start_node_type": "TextChunk",
            "start_node_property": "chunk_id",
            "end_node_type": "Entity",
            "end_node_property": "name",
            "relationship_type": "MENTIONS",
            "properties": {"frequency": 1}
        }
    ]
    graph_builder.add_custom_relationships(custom_relationships)

    neo4j_conn.close()