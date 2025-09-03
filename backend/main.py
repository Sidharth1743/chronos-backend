
import os
from dotenv import load_dotenv
from services.ocr import OCRProcessor
from services.semantic import SemanticProcessor
from services.pinata import PinataUploader
from services.ner import LangChainKnowledgeGraphAgent, export_to_json, validate_graph_element
from services.neo4j_ops import KnowledgeGraphPipeline

load_dotenv()

def main():
    # Load API keys and config from .env
    openai_api_key = os.getenv("OPENAI_API_KEY")
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    pinata_api_key = os.getenv("PINATA_API_KEY")
    pinata_secret_api_key = os.getenv("PINATA_SECRET_API_KEY")

    # Step 1: OCR
    ocr_input_file = os.getenv("OCR_INPUT_FILE", "../doc.pdf")
    ocr_processor = OCRProcessor()
    ocr_text = ocr_processor.extract_text(ocr_input_file)

    # Step 2: Semantic
    semantic_processor = SemanticProcessor()
    semantic_output = semantic_processor.process(ocr_text)
    semantic_file = "results/semantic_output.txt"
    with open(semantic_file, "w", encoding="utf-8") as f:
        f.write(semantic_output)

    # Step 3: Pinata
    pinata_uploader = PinataUploader(api_key=pinata_api_key, secret_api_key=pinata_secret_api_key)
    pinata_link = pinata_uploader.upload_document(semantic_file)
    pinata_link_file = "results/pinata_link.txt"
    with open(pinata_link_file, "w", encoding="utf-8") as f:
        f.write(pinata_link)

    # Step 4: NER
    agent = LangChainKnowledgeGraphAgent()
    graph_elements = agent.extract_from_text(semantic_output)
    if not graph_elements:
        print("No graph elements extracted.")
        return
    if len(graph_elements) > 1:
        merged = agent.merge_graph_elements(graph_elements)
    else:
        merged = graph_elements[0]
    if validate_graph_element(merged):
        json_data = export_to_json(merged)
        ner_json_file = "results/ner_output.json"
        import json
        with open(ner_json_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
        print(f"Exported NER graph to {ner_json_file}")
    else:
        print("Invalid graph element structure.")
        return

    # Step 5: Neo4j
    pipeline = KnowledgeGraphPipeline(
        openai_api_key=openai_api_key,
        neo4j_uri=neo4j_uri,
        neo4j_username=neo4j_username,
        neo4j_password=neo4j_password
    )
    success = pipeline.store_in_neo4j([merged])
    if success:
        print(f"Stored {len(merged.nodes)} nodes and {len(merged.relationships)} relationships in Neo4j.")
    stats = pipeline.get_statistics()
    print(f"Knowledge graph contains {stats['nodes']} nodes and {stats['relationships']} relationships.")

if __name__ == "__main__":
    main()