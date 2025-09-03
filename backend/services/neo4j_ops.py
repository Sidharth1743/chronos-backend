from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain_neo4j import GraphCypherQAChain
from .ner import Node, Relationship, GraphElement  # Import from ner.py
from dotenv import load_dotenv
import requests

load_dotenv() 

class KnowledgeGraphPipeline:
    """Pipeline for Neo4j operations with Tamil historical spine science knowledge graph."""
    
    def __init__(
        self,
        openai_api_key: str,
        neo4j_uri: str,
        neo4j_username: str,
        neo4j_password: str
    ):
        """
        Initialize the Neo4j pipeline.
        
        Args:
            openai_api_key: OpenAI API key
            neo4j_uri: Neo4j database URI
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
        """
        # Initialize LLM
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-4o",
            temperature=0.1
        )
        
        # Initialize Neo4j connection
        self.graph = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password
        )

    def store_in_neo4j(self, graph_elements: List[GraphElement]) -> bool:
        """
        Store extracted graph elements in Neo4j database.
        
        Args:
            graph_elements: List of GraphElement objects to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            for element in graph_elements:
                # Create nodes
                for node in element.nodes:
                    cypher = f"""
                    MERGE (n:{node.type} {{id: $id}})
                    SET n += $properties
                    """
                    self.graph.query(
                        cypher, 
                        {"id": node.id, "properties": node.properties}
                    )
                
                # Create relationships
                for rel in element.relationships:
                    cypher = f"""
                    MATCH (a:{rel.subj.type} {{id: $subj_id}})
                    MATCH (b:{rel.obj.type} {{id: $obj_id}})
                    MERGE (a)-[r:{rel.type.upper()}]->(b)
                    SET r += $properties
                    """
                    rel_props = rel.properties.copy()
                    if rel.timestamp:
                        rel_props['timestamp'] = rel.timestamp
                    
                    self.graph.query(
                        cypher,
                        {
                            "subj_id": rel.subj.id,
                            "obj_id": rel.obj.id,
                            "properties": rel_props
                        }
                    )
            return True
            
        except Exception as e:
            print(f"Error storing in Neo4j: {e}")
            return False

    def query_knowledge_base(self, question: str) -> str:
        """
        Query the knowledge base with natural language.
        
        Args:
            question: Natural language question
            
        Returns:
            Answer from the knowledge graph
        """
        qa_chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            verbose=True,
            allow_dangerous_requests=True
        )
        return qa_chain.run(question)

    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about the knowledge graph."""
        node_count_query = "MATCH (n) RETURN count(n) as node_count"
        rel_count_query = "MATCH ()-[r]->() RETURN count(r) as rel_count"
        
        node_result = self.graph.query(node_count_query)
        rel_result = self.graph.query(rel_count_query)
        
        return {
            "nodes": node_result[0]['node_count'] if node_result else 0,
            "relationships": rel_result[0]['rel_count'] if rel_result else 0
        }

    def get_graph_schema(self) -> str:
        """Get the current graph schema."""
        return self.graph.get_schema