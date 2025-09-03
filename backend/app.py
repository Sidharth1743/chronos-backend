from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory, jsonify
import os
from dotenv import load_dotenv
from services.ocr import OCRProcessor
from services.semantic import SemanticProcessor
from services.pinata import PinataUploader
from services.ner import LangChainKnowledgeGraphAgent, export_to_json, validate_graph_element
from services.neo4j_ops import KnowledgeGraphPipeline
import json

# Load environment variables
load_dotenv()

app = Flask(__name__, 
           template_folder='../templates',  # Templates are in parent directory
           static_folder='../static')       # Static files are in parent directory

app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your_secret_key_here')
UPLOAD_FOLDER = '../uploads'  # Upload folder in parent directory
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create required directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('results', exist_ok=True)

# Initialize services
openai_api_key = os.getenv("OPENAI_API_KEY")
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")
pinata_api_key = os.getenv("PINATA_API_KEY")
pinata_secret_api_key = os.getenv("PINATA_SECRET_API_KEY")

ocr_processor = OCRProcessor()
semantic_processor = SemanticProcessor()
pinata_uploader = PinataUploader(api_key=pinata_api_key, secret_api_key=pinata_secret_api_key)
ner_agent = LangChainKnowledgeGraphAgent()
knowledge_graph_pipeline = KnowledgeGraphPipeline(
    openai_api_key=openai_api_key,
    neo4j_uri=neo4j_uri,
    neo4j_username=neo4j_username,
    neo4j_password=neo4j_password
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_pdf(filepath):
    try:
        # Step 1: OCR
        ocr_text = ocr_processor.extract_text(filepath)

        # Step 2: Semantic
        semantic_output = semantic_processor.process(ocr_text)
        semantic_file = "results/semantic_output.txt"
        with open(semantic_file, "w", encoding="utf-8") as f:
            f.write(semantic_output)

        # Step 3: Pinata
        pinata_link = pinata_uploader.upload_document(semantic_file)
        pinata_link_file = "results/pinata_link.txt"
        with open(pinata_link_file, "w", encoding="utf-8") as f:
            f.write(pinata_link)

        # Step 4: NER
        graph_elements = ner_agent.extract_from_text(semantic_output)
        if not graph_elements:
            raise Exception("No graph elements extracted.")
            
        if len(graph_elements) > 1:
            merged = ner_agent.merge_graph_elements(graph_elements)
        else:
            merged = graph_elements[0]

        if validate_graph_element(merged):
            json_data = export_to_json(merged)
            ner_json_file = "results/ner_output.json"
            with open(ner_json_file, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2)
        else:
            raise Exception("Invalid graph element structure.")

        # Step 5: Neo4j
        success = knowledge_graph_pipeline.store_in_neo4j([merged])
        if not success:
            raise Exception("Failed to store in Neo4j")

        stats = knowledge_graph_pipeline.get_statistics()
        
        return {
            'pinata_link': pinata_link,
            'nodes': stats['nodes'],
            'relationships': stats['relationships'],
            'success': True
        }
    except Exception as e:
        print(f"Error in process_pdf: {str(e)}")
        return {'error': str(e), 'success': False}

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            result = process_pdf(filepath)
            
            if result['success']:
                flash('File processed successfully')
                return redirect(url_for('view_pdf', filename=filename))
            else:
                flash(f'Error processing file: {result.get("error", "Unknown error")}')
                return redirect(request.url)
        else:
            flash('Please upload only PDF files')
    
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, mimetype='application/pdf')

@app.route('/view/<filename>')
def view_pdf(filename):
    try:
        # Get pinata link
        with open('results/pinata_link.txt', 'r') as f:
            pinata_link = f.read().strip()

        # Get graph data directly from Neo4j
        neo4j_graph = knowledge_graph_pipeline.graph
        
        # Fetch all nodes with their labels and properties
        nodes_query = """
        MATCH (n)
        WITH n, labels(n) as nodeLabels, properties(n) as props
        RETURN collect({
            id: n.id,
            labels: nodeLabels,
            properties: props
        }) as nodes
        """
        
        # Fetch all relationships
        rels_query = """
        MATCH (source)-[r]->(target)
        WITH source, r, target, 
             labels(source) as sourceLabels,
             labels(target) as targetLabels,
             properties(r) as relProps
        RETURN collect({
            source: source.id,
            target: target.id,
            type: type(r),
            properties: relProps
        }) as relationships
        """
        
        # Execute queries
        nodes_result = neo4j_graph.query(nodes_query)
        rels_result = neo4j_graph.query(rels_query)
        
        # Transform nodes
        nodes = []
        node_ids = set()
        
        for node in nodes_result[0]['nodes']:
            node_ids.add(node['id'])
            nodes.append({
                'id': node['id'],
                'labels': node['labels'],
                'properties': node.get('properties', {}),
                'title': node.get('properties', {}).get('name', node['id']).replace('_', ' ').title()
            })
        
        # Transform relationships
        links = []
        for rel in rels_result[0]['relationships']:
            source_id = rel['source']
            target_id = rel['target']
            
            if source_id not in node_ids or target_id not in node_ids:
                continue
                
            links.append({
                'source': source_id,
                'target': target_id,
                'type': rel['type'],
                'properties': rel.get('properties', {})
            })
        
        # Get graph statistics
        stats = knowledge_graph_pipeline.get_statistics()
        
        # Prepare the complete graph data
        graph_data = {
            'nodes': nodes,
            'links': links,
            'stats': {
                'totalNodes': stats['nodes'],
                'totalRelationships': stats['relationships']
            }
        }
        
        return render_template('view_pdf.html',
                             filename=filename,
                             graph_data=json.dumps(graph_data),
                             pinata_link=pinata_link)
    except Exception as e:
        print(f"Error in view_pdf: {str(e)}")
        return render_template('view_pdf.html',
                             filename=filename,
                             graph_data=json.dumps({
                                 'nodes': [],
                                 'links': [],
                                 'stats': {
                                     'totalNodes': 0,
                                     'totalRelationships': 0
                                 }
                             }),
                             pinata_link='')

if __name__ == '__main__':
    app.run(debug=True)
