from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
# from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import requests
from .umls import UMLSService
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv() 

@dataclass
class Node:
    """Represents a node in the knowledge graph."""
    id: str
    type: str
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


@dataclass
class Relationship:
    """Represents a relationship between two nodes."""
    subj: Node
    obj: Node
    type: str
    properties: Dict[str, Any] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


@dataclass
class GraphElement:
    """Container for nodes and relationships extracted from content."""
    nodes: List[Node]
    relationships: List[Relationship]
    source: Any = None


class GraphExtractionParser(BaseOutputParser):
    """Custom parser to extract nodes and relationships from LLM output."""
    
    def parse(self, text: str) -> GraphElement:
        """Parse the LLM output into GraphElement."""
        nodes = {}
        relationships = []

        # Regular expressions to extract nodes and relationships
        node_pattern = r"Node\(id='(.*?)', type='(.*?)'\)"
        rel_pattern = (r"Relationship\(subj=Node\(id='(.*?)', type='(.*?)'\), "
                      r"obj=Node\(id='(.*?)', type='(.*?)'\), "
                      r"type='(.*?)'(?:, timestamp='(.*?)')?\)")

        # Extract nodes
        for match in re.finditer(node_pattern, text):
            node_id, node_type = match.groups()
            if node_id not in nodes:
                node = Node(id=node_id, type=node_type)
                nodes[node_id] = node

        # Extract relationships
        for match in re.finditer(rel_pattern, text):
            groups = match.groups()
            if len(groups) == 6:
                subj_id, subj_type, obj_id, obj_type, rel_type, timestamp = groups
            else:
                subj_id, subj_type, obj_id, obj_type, rel_type = groups
                timestamp = None
            
            # Create nodes if they don't exist
            if subj_id not in nodes:
                nodes[subj_id] = Node(id=subj_id, type=subj_type)
            if obj_id not in nodes:
                nodes[obj_id] = Node(id=obj_id, type=obj_type)
            
            subj = nodes[subj_id]
            obj = nodes[obj_id]
            relationship = Relationship(
                subj=subj,
                obj=obj,
                type=rel_type,
                timestamp=timestamp
            )
            relationships.append(relationship)

        return GraphElement(
            nodes=list(nodes.values()),
            relationships=relationships,
            source=text
        )


class LangChainKnowledgeGraphAgent:
    """LangChain-based Knowledge Graph extraction agent for Tamil historical spine science texts."""
    
    def __init__(
        self,
        llm: Optional[ChatGoogleGenerativeAI] = None,
        custom_prompt: Optional[str] = None
    ):
        """
        Initialize the Knowledge Graph Agent.
        
        Args:
            llm: ChatOpenAI model instance
            custom_prompt: Custom extraction prompt (optional)
        """
        try:
            self.llm = llm or ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.1,
                max_tokens=5000
            )
            
            # Default extraction prompt
            self.extraction_prompt = custom_prompt or self._get_default_prompt()
            
            # Create prompt template
            self.prompt_template = PromptTemplate(
                input_variables=["content"],
                template=self.extraction_prompt
            )
            
            # Create parser
            self.parser = GraphExtractionParser()
            
            # Create extraction chain
            self.extraction_chain = self.prompt_template | self.llm | self.parser
            
            # Text splitter for large documents
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            
            # Initialize UMLS service
            try:
                self.umls_service = UMLSService()
            except Exception as e:
                print(f"Warning: Could not initialize UMLS service: {e}. UMLS enrichment will be skipped.")
                self.umls_service = None
                
        except Exception as e:
            print(f"Error initializing LangChainKnowledgeGraphAgent: {e}")
            raise

    def _get_default_prompt(self) -> str:
        """Get the default extraction prompt with HypE integration."""
        return """
You are tasked with extracting entities (nodes) and relationships from Tamil language texts related to historical spine science and traditional medicine, then structuring them into Node and Relationship objects. The input will be in Tamil, but your extracted nodes and relationships should be translated into English. Integrate the Hypothesis and Evidence Ontology (HypE) by including Hypothesis and Evidence node types and their relationships.

Content Extraction:
- Process the input content and identify entities mentioned within it.
- Focus on entities related to historical spine treatments, observations, medical concepts, and hypotheses/evidence.
- If the content is in Tamil, ensure proper translation to English equivalents.

Node Extraction:
For each identified entity, create a Node object.
Each Node object should have a unique identifier (id) and a type (type).
Node types should be one of the following:
- ClinicalObservation (signs, symptoms, disease presentations, e.g., prevalence_sbo_in_ds_11_9_percent)
- TherapeuticOutcome (treatment responses, recovery patterns)
- ContextualFactor (environmental, behavioral, constitutional factors, e.g., male_to_female_ratio_is_sbo)
- MechanisticConcept (traditional explanatory models, processes, e.g., sbo_occulta)
- TherapeuticApproach (interventions, remedies, methods)
- SourceText (reference to original documents or authors, e.g., article_authors)
- PathophysiologicalPattern (imbalances like Qi stagnation, cold-damp obstruction)
- DiagnosticMethod (tongue diagnosis, pulse reading, imaging)
- Biomarker (lab values, imaging results, pulse findings)
- PractitionerRole (herbalist, acupuncturist, orthopedic specialist)
- AnatomicalStructure (spine, vertebrae, nervous pathways)
- HealthOutcome (restored mobility, pain reduction, improved sleep)
- DosageForm (decoction, acupuncture session, topical paste)
- TemporalPhase (acute, subacute, chronic, post-therapy)
- Hypothesis (hypothesized relationships, e.g., sbo_increases_ds_risk)
- Evidence (data supporting hypotheses, e.g., prevalence studies like prevalence_sbo_in_ds_11_9_percent)

Relationship Extraction:
Identify relationships between extracted entities in the content.
For each relationship, create a Relationship object with types:
- co_occurs_with, preceded_by, followed_by, modified_by, responds_to
- associated_with, results_in, described_in, contradicts, corroborates
- indicates, measured_by, administered_by, targets, progresses_to
- contraindicated_in, expressed_during, documented_by
- supports (Evidence supports Hypothesis, e.g., prevalence_sbo_in_ds_11_9_percent supports sbo_increases_ds_risk)
- cites (Evidence or Hypothesis cites SourceText, e.g., prevalence_sbo_in_ds_11_9_percent cites article_authors)

Output Format:
Provide ONLY the Node and Relationship objects in this exact format:

Nodes:
Node(id='unique_identifier', type='NodeType')

Relationships:
Relationship(subj=Node(id='subject_id', type='SubjectType'), obj=Node(id='object_id', type='ObjectType'), type='relationship_type')

Example Content:
"Spina bifida occulta (SBO) is associated with degenerative spondylolisthesis (DS) with a prevalence of 11.9%. Studies suggest SBO may increase the risk of DS."

Example Output:
Nodes:
Node(id='sbo_occulta', type='MechanisticConcept')
Node(id='degenerative_spondylolisthesis', type='ClinicalObservation')
Node(id='prevalence_sbo_in_ds_11_9_percent', type='Evidence')
Node(id='sbo_increases_ds_risk', type='Hypothesis')
Node(id='article_authors', type='SourceText')

Relationships:
Relationship(subj=Node(id='sbo_occulta', type='MechanisticConcept'), obj=Node(id='degenerative_spondylolisthesis', type='ClinicalObservation'), type='associated_with')
Relationship(subj=Node(id='prevalence_sbo_in_ds_11_9_percent', type='Evidence'), obj=Node(id='sbo_increases_ds_risk', type='Hypothesis'), type='supports')
Relationship(subj=Node(id='prevalence_sbo_in_ds_11_9_percent', type='Evidence'), obj=Node(id='article_authors', type='SourceText'), type='cites')

Content to process:
{content}
"""

    def extract_from_text(
        self, 
        content: str, 
        chunk_large_text: bool = True
    ) -> List[GraphElement]:
        """
        Extract knowledge graph elements from text content with HypE integration.
        
        Args:
            content: Input text content (can be in Tamil)
            chunk_large_text: Whether to split large texts into chunks
            
        Returns:
            List of GraphElement objects containing extracted nodes and relationships
        """
        if chunk_large_text and len(content) > 2000:
            chunks = self.text_splitter.split_text(content)
            results = []
            
            for chunk in chunks:
                try:
                    result = self.extraction_chain.invoke({"content": chunk})
                    result = self._add_hype_nodes_and_rels(result)  # Add HypE nodes/rels
                    if result.nodes or result.relationships:
                        results.append(result)
                except Exception as e:
                    print(f"Error processing chunk: {e}")
                    continue
            
            return results
        else:
            try:
                result = self.extraction_chain.invoke({"content": content})
                result = self._add_hype_nodes_and_rels(result)  # Add HypE nodes/rels
                return [result] if result.nodes or result.relationships else []
            except Exception as e:
                print(f"Error processing content: {e}")
                return []

    def _add_hype_nodes_and_rels(self, graph_element: GraphElement) -> GraphElement:
        """
        Add HypE-specific nodes and relationships to the graph element.
        Also enriches nodes with UMLS concepts if UMLS service is available.
        
        Args:
            graph_element: GraphElement object to enhance
            
        Returns:
            Enhanced GraphElement with HypE nodes and relationships, and UMLS concepts
        """
        new_nodes = []
        new_relationships = []
        
        # Process each node
        for node in graph_element.nodes:
            # Add UMLS concepts if service is available
            if self.umls_service:
                node_term = node.id.replace('_', ' ')
                try:
                    umls_data = self.umls_service.get_umls_concept(node_term)
                    if umls_data:
                        if not node.properties:
                            node.properties = {}
                        node.properties.update({
                            'umls_cui': umls_data['cui'],
                            'umls_semantic_type': umls_data['semantic_type'],
                            'umls_preferred_name': umls_data['preferred_name']
                        })
                except Exception as e:
                    print(f"Warning: UMLS lookup failed for term '{node_term}': {e}")
            
            # Re-label ClinicalObservation as Evidence and add Hypothesis nodes
            if node.type == "ClinicalObservation":
                node.type = "Evidence"  # HypE re-labeling
            
            # Infer Hypothesis nodes for terms suggesting causality
            if any(term in node.id.lower() for term in ["increase", "risk", "cause", "effect"]):
                hypothesis_node = Node(
                    id=f"{node.id}_hypothesis",
                    type="Hypothesis",
                    properties={
                        "source": "langchain_extracted",
                        "original_term": node_term
                    }
                )
                # Try to get UMLS concept for hypothesis
                hyp_umls_data = self.umls_service.get_umls_concept(node_term + " hypothesis")
                if hyp_umls_data:
                    hypothesis_node.properties.update({
                        'umls_cui': hyp_umls_data['cui'],
                        'umls_semantic_type': hyp_umls_data['semantic_type'],
                        'umls_preferred_name': hyp_umls_data['preferred_name']
                    })
                new_nodes.append(hypothesis_node)
                new_relationships.append(Relationship(
                    subj=node,
                    obj=hypothesis_node,
                    type="supports",
                    properties={"source": "langchain_extracted"}
                ))
        
        # Add cites relationships for Evidence to SourceText
        for node in graph_element.nodes:
            if node.type == "Evidence":
                for source_node in graph_element.nodes:
                    if source_node.type == "SourceText":
                        new_relationships.append(Relationship(
                            subj=node,
                            obj=source_node,
                            type="cites",
                            properties={"source": "langchain_extracted"}
                        ))
        
        graph_element.nodes.extend(new_nodes)
        graph_element.relationships.extend(new_relationships)
        return graph_element

    def extract_from_document(self, document: Document) -> List[GraphElement]:
        """
        Extract knowledge graph elements from a LangChain Document.
        
        Args:
            document: LangChain Document object
            
        Returns:
            List of GraphElement objects
        """
        return self.extract_from_text(document.page_content)

    def merge_graph_elements(self, elements_list: List[GraphElement]) -> GraphElement:
        """
        Merge multiple GraphElement objects into one, removing duplicates.
        
        Args:
            elements_list: List of GraphElement objects to merge
            
        Returns:
            Merged GraphElement object
        """
        all_nodes = {}
        all_relationships = []
        
        for element in elements_list:
            # Merge nodes (avoid duplicates by ID)
            for node in element.nodes:
                if node.id not in all_nodes:
                    all_nodes[node.id] = node
            
            # Add relationships (with duplicate checking)
            for rel in element.relationships:
                exists = any(
                    existing.subj.id == rel.subj.id and
                    existing.obj.id == rel.obj.id and
                    existing.type == rel.type
                    for existing in all_relationships
                )
                if not exists:
                    all_relationships.append(rel)
        
        return GraphElement(
            nodes=list(all_nodes.values()),
            relationships=all_relationships,
            source="merged_elements"
        )


def export_to_json(graph_element: GraphElement) -> Dict[str, Any]:
    """Export GraphElement to JSON format."""
    return {
        "nodes": [
            {
                "id": node.id,
                "type": node.type,
                "properties": node.properties
            }
            for node in graph_element.nodes
        ],
        "relationships": [
            {
                "subject": {"id": rel.subj.id, "type": rel.subj.type},
                "object": {"id": rel.obj.id, "type": rel.obj.type},
                "type": rel.type,
                "timestamp": rel.timestamp,
                "properties": rel.properties
            }
            for rel in graph_element.relationships
        ]
    }


def validate_graph_element(graph_element: GraphElement) -> bool:
    """Validate the structure of a GraphElement."""
    if not isinstance(graph_element, GraphElement):
        return False
    
    # Validate nodes
    for node in graph_element.nodes:
        if not isinstance(node, Node) or not node.id or not node.type:
            return False
    
    # Validate relationships
    for rel in graph_element.relationships:
        if not isinstance(rel, Relationship) or not rel.type:
            return False
        if not isinstance(rel.subj, Node) or not isinstance(rel.obj, Node):
            return False
    
    return True