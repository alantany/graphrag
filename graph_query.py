from neo4j import GraphDatabase
from typing import List, Dict, Any
from openai import OpenAI

class GraphQuery:
    def __init__(self, neo4j_connection):
        self.neo4j_connection = neo4j_connection
        self.client = OpenAI(
            api_key="sk-1pUmQlsIkgla3CuvKTgCrzDZ3r0pBxO608YJvIHCN18lvOrn",
            base_url="https://api.chatanywhere.tech/v1"
        )

    def search_by_keywords(self, keywords: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """
        基于关键词搜索相关文档块
        """
        query = """
        MATCH (n:TextChunk)
        WHERE any(keyword IN $keywords WHERE n.content CONTAINS keyword)
        RETURN n.chunk_id AS chunk_id, n.file_name AS file_name, n.content AS content
        LIMIT $limit
        """
        result = self.neo4j_connection.query(query, parameters={'keywords': keywords, 'limit': limit})
        return [dict(record) for record in result]

    def get_document_chunks(self, file_name: str) -> List[Dict[str, Any]]:
        """
        查找特定文档的所有相关块
        """
        query = """
        MATCH (n:TextChunk {file_name: $file_name})
        RETURN n.chunk_id AS chunk_id, n.content AS content
        ORDER BY n.chunk_id
        """
        result = self.neo4j_connection.query(query, parameters={'file_name': file_name})
        return [dict(record) for record in result]

    def find_document_relationships(self, file_name: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """
        发现文档之间的关系
        """
        query = """
        MATCH (start:TextChunk {file_name: $file_name})
        CALL apoc.path.subgraphNodes(start, {
            relationshipFilter: "NEXT",
            maxLevel: $max_depth
        })
        YIELD node
        WHERE node.file_name <> $file_name
        RETURN DISTINCT node.file_name AS related_file, count(node) AS chunk_count
        ORDER BY chunk_count DESC
        """
        result = self.neo4j_connection.query(query, parameters={'file_name': file_name, 'max_depth': max_depth})
        return [dict(record) for record in result]

    def neo4j_qa(self, query: str) -> tuple:
        """
        使用Neo4j进行问答
        """
        # 使用关键词搜索相关文本块
        keywords = query.split()  # 简单地将查询分割为关键词
        cypher_query = """
        MATCH (n:TextChunk)
        WHERE any(keyword IN $keywords WHERE n.content CONTAINS keyword)
        RETURN n.chunk_id AS chunk_id, n.file_name AS file_name, n.content AS content
        LIMIT 3
        """
        result = self.neo4j_connection.query(cypher_query, parameters={'keywords': keywords})
        
        if not result:
            return "没有找到相关信息。", [], ""

        context = "\n".join([record['content'] for record in result])
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一位有帮助的助手。请根据给定的上下文回答问题。始终使用中文回答，无论问题是什么语言。在回答之后，请务必提供一个最相关的原文摘录，以'相关原文：'为前缀。"},
                {"role": "user", "content": f"上下文: {context}\n\n问题: {query}\n\n请提供你的回答然后在回答后面附上相关的原文摘录，以'相关原文：'为前缀。"}
            ]
        )
        answer = response.choices[0].message.content
        
        # 处理回答格式
        if "相关原文：" in answer:
            answer_parts = answer.split("相关原文：", 1)
            main_answer = answer_parts[0].strip()
            relevant_excerpt = answer_parts[1].strip()
        else:
            main_answer = answer.strip()
            relevant_excerpt = result[0]['content'][:200] + "..."  # 使用第一个结果的前200个字符
        
        sources = [(record['file_name'], record['chunk_id']) for record in result]
        
        return main_answer, sources, relevant_excerpt

# 使用示例
if __name__ == "__main__":
    from neo4j_operations import get_neo4j_connection, LOCAL_CONFIG

    neo4j_conn = get_neo4j_connection(LOCAL_CONFIG)
    graph_query = GraphQuery(neo4j_conn)

    # 测试关键词搜索
    keywords = ["医疗", "患者"]
    results = graph_query.search_by_keywords(keywords)
    print("关键词搜索结果:", results)

    # 测试获取文档块
    file_name = "example.pdf"
    chunks = graph_query.get_document_chunks(file_name)
    print(f"{file_name} 的文档块:", chunks)

    # 测试文档关系查询
    relationships = graph_query.find_document_relationships(file_name)
    print(f"{file_name} 的相关文档:", relationships)

    neo4j_conn.close()