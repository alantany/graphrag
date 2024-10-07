from langchain import OpenAI, LLMChain
from langchain.prompts import PromptTemplate
from neo4j import GraphDatabase

def process_data(content):
    # 使用LangChain处理文本
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["content"],
        template="从以下内容中提取实体和关系:\n{content}\n实体和关系:"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(content)
    
    # 将结果存入Neo4j
    save_to_neo4j(result)
    
    return result

def save_to_neo4j(data):
    uri = "neo4j://localhost:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))
    
    with driver.session() as session:
        # 这里需要根据实际的数据结构来编写Cypher查询
        session.run("CREATE (n:Entity {name: $name})", name="示例实体")
    
    driver.close()