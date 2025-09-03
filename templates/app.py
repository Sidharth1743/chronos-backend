from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory, jsonify
import os
from dotenv import load_dotenv
from backend.services.ocr import OCRProcessor
from backend.services.semantic import SemanticProcessor
from backend.services.pinata import PinataUploader
from backend.services.ner import LangChainKnowledgeGraphAgent, export_to_json, validate_graph_element
from backend.services.neo4j_ops import KnowledgeGraphPipeline
import json

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your_secret_key_here')
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}  # Restricting to PDF files only
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create required directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('backend/results', exist_ok=True)

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
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_pdf(filepath):
    try:
        # Step 1: OCR
        ocr_text = ocr_processor.extract_text(filepath)

        # Step 2: Semantic
        semantic_output = semantic_processor.process(ocr_text)
        semantic_file = "backend/results/semantic_output.txt"
        with open(semantic_file, "w", encoding="utf-8") as f:
            f.write(semantic_output)

        # Step 3: Pinata
        pinata_link = pinata_uploader.upload_document(semantic_file)
        pinata_link_file = "backend/results/pinata_link.txt"
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
            ner_json_file = "backend/results/ner_output.json"
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
            
            # Process the PDF through our pipeline
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
        with open('backend/results/pinata_link.txt', 'r') as f:
            pinata_link = f.read().strip()
            
        # Get graph data directly from Neo4j
        neo4j_graph = knowledge_graph_pipeline.graph
        
        # Fetch all nodes
        nodes_query = """
        MATCH (n)
        RETURN collect({
            id: n.id,
            type: labels(n)[0],
            properties: properties(n)
        }) as nodes
        """
        nodes_result = neo4j_graph.query(nodes_query)
        nodes_data = nodes_result[0]['nodes'] if nodes_result else []
        
        # Fetch all relationships
        rels_query = """
        MATCH (source)-[r]->(target)
        RETURN collect({
            source: source.id,
            target: target.id,
            type: type(r),
            properties: properties(r)
        }) as relationships
        """
        rels_result = neo4j_graph.query(rels_query)
        rels_data = rels_result[0]['relationships'] if rels_result else []
        
        # Transform data for visualization
        nodes = []
        links = []
        node_ids = set()
        
        # Process nodes
        for node in nodes_data:
            # Process node properties
            properties = node.get('properties', {})
            
            # Extract UMLS data if available
            umls_data = {
                'cui': properties.get('umls_cui'),
                'semantic_type': properties.get('umls_semantic_type'),
                'preferred_name': properties.get('umls_preferred_name')
            }
            
            # Remove UMLS data from general properties
            for k in ['umls_cui', 'umls_semantic_type', 'umls_preferred_name']:
                if k in properties:
                    del properties[k]
            node_ids.add(node['id'])
            nodes.append({
                'id': node['id'],
                'labels': [node['type']],
                'properties': node.get('properties', {}),
                'title': node['id'].replace('_', ' ').title()  # Make ID readable
            })
            
        # Then process relationships from Neo4j data
        for rel in rels_data:
            source_id = rel['source']
            target_id = rel['target']
            
            # Add any nodes that might only appear in relationships
            if source_id not in node_ids:
                node_ids.add(source_id)
                nodes.append({
                    'id': source_id,
                    'labels': ['Unknown'],  # Default label since we don't have type info
                    'properties': {},
                    'title': source_id.replace('_', ' ').title()
                })
            
            if target_id not in node_ids:
                node_ids.add(target_id)
                nodes.append({
                    'id': target_id,
                    'labels': ['Unknown'],  # Default label since we don't have type info
                    'properties': {},
                    'title': target_id.replace('_', ' ').title()
                })
            
        # Process relationships
        for rel in rels_data:
            links.append({
                'source': rel['source'],
                'target': rel['target'],
                'type': rel['type'],
                'properties': rel.get('properties', {})
            })
            
        return render_template('view_pdf.html', 
                             filename=filename,
                             graph_data=json.dumps({'nodes': nodes, 'links': links}),
                             pinata_link=pinata_link)
    except Exception as e:
        print(f"Error processing graph data: {str(e)}")
        return render_template('view_pdf.html', 
                             filename=filename,
                             graph_data=json.dumps({'nodes': [], 'links': []}),
                             pinata_link='')

@app.route('/get_graph_data')
def get_graph_data():
    try:
        with open('backend/results/ner_output.json', 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_pinata_link')
def get_pinata_link():
    try:
        with open('backend/results/pinata_link.txt', 'r') as f:
            link = f.read().strip()
        return jsonify({'link': link})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

if __name__ == '__main__':
    app.run(debug=True)