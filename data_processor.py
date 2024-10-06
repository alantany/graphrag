import numpy as np
import faiss
from typing import Dict, List, Any

class DataProcessor:
    def __init__(self):
        pass

    def prepare_data_for_neo4j(self, file_indices: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        从file_indices准备数据以导入Neo4j
        """
        data = []
        for file_name, (chunks, index) in file_indices.items():
            vectors = self.get_vectors_from_faiss(index)
            for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
                data.append({
                    "file_name": file_name,
                    "chunk_id": f"{file_name}_{i}",
                    "content": chunk,
                    "embedding": vector.tolist()
                })
        return data

    def get_vectors_from_faiss(self, index: faiss.IndexFlatL2) -> np.ndarray:
        """
        从FAISS索引中提取向量
        """
        return index.reconstruct_n(0, index.ntotal)

    def create_relationships(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        基于文本块之间的关系创建关系数据
        """
        relationships = []
        for i in range(len(data) - 1):
            if data[i]["file_name"] == data[i + 1]["file_name"]:
                relationships.append({
                    "start_node": data[i]["chunk_id"],
                    "end_node": data[i + 1]["chunk_id"],
                    "type": "NEXT",
                    "properties": {"file_name": data[i]["file_name"]}
                })
        return relationships

    def process_file_indices(self, file_indices: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理file_indices，准备节点和关系
        """
        node_data = self.prepare_data_for_neo4j(file_indices)
        relationship_data = self.create_relationships(node_data)
        
        return {
            "nodes": node_data,
            "relationships": relationship_data
        }

def export_to_neo4j(file_indices: Dict[str, Any], neo4j_connection):
    """
    将处理后的数据导出到Neo4j
    """
    processor = DataProcessor()
    processed_data = processor.process_file_indices(file_indices)

    # 创建节点
    for node in processed_data["nodes"]:
        neo4j_connection.create_node("TextChunk", {
            "chunk_id": node["chunk_id"],
            "file_name": node["file_name"],
            "content": node["content"],
            "embedding": str(node["embedding"])  # Neo4j可能不支持直接存储列表，所以转换为字符串
        })

    # 创建关系
    for rel in processed_data["relationships"]:
        neo4j_connection.create_relationship(
            "TextChunk", "chunk_id", "TextChunk", "chunk_id",
            rel["type"], {"file_name": rel["properties"]["file_name"]}
        )

    return len(processed_data["nodes"]), len(processed_data["relationships"])

def query_imported_data(neo4j_connection):
    """
    查询最近导入到Neo4j的数据样本
    """
    sample_query = """
    MATCH (n:TextChunk)
    RETURN n
    ORDER BY n.chunk_id DESC
    LIMIT 5
    """
    samples = neo4j_connection.query(sample_query)
    return sample_query, samples