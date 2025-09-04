from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory, jsonify, session
import os
import tempfile
import json
import gzip
import base64
from dotenv import load_dotenv
from services.ocr import OCRProcessor
from services.semantic import SemanticProcessor
from services.pinata import PinataUploader
from services.ner import LangChainKnowledgeGraphAgent, export_to_json, validate_graph_element
from services.neo4j_ops import KnowledgeGraphPipeline

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
pinata_api_key = os.getenv("PINATA_API_KEY")
pinata_secret_api_key = os.getenv("PINATA_SECRET_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

ocr_processor = OCRProcessor()
semantic_processor = SemanticProcessor()
pinata_uploader = PinataUploader(api_key=pinata_api_key, secret_api_key=pinata_secret_api_key)
ner_agent = LangChainKnowledgeGraphAgent()

# Neo4j is now optional since we use session-based storage
knowledge_graph_pipeline = None
try:
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    if neo4j_uri and neo4j_username and neo4j_password:
        knowledge_graph_pipeline = KnowledgeGraphPipeline(
            google_api_key=google_api_key,
            neo4j_uri=neo4j_uri,
            neo4j_username=neo4j_username,
            neo4j_password=neo4j_password
        )
        print("Neo4j connection initialized (optional)")
    else:
        print("Neo4j credentials not found - using session-based storage only")
except Exception as e:
    print(f"Neo4j initialization failed - using session-based storage only: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def compress_data(data):
    """Compress data for session storage to avoid cookie size limits."""
    try:
        json_str = json.dumps(data)
        compressed = gzip.compress(json_str.encode('utf-8'))
        return base64.b64encode(compressed).decode('utf-8')
    except Exception as e:
        print(f"Error compressing data: {e}")
        return None

def decompress_data(compressed_data):
    """Decompress data from session storage."""
    try:
        if not compressed_data:
            return None
        compressed = base64.b64decode(compressed_data.encode('utf-8'))
        json_str = gzip.decompress(compressed).decode('utf-8')
        return json.loads(json_str)
    except Exception as e:
        print(f"Error decompressing data: {e}")
        return None

def store_graph_data(nodes, relationships):
    """Store graph data in session with compression and size optimization."""
    # Important properties to keep for medical/scientific knowledge graphs
    important_properties = [
        'name', 'type', 'id', 'source',
        'umls_cui', 'umls_preferred_name', 'umls_semantic_type',
        'description', 'category', 'label', 'title'
    ]
    
    # Optimize data while preserving important medical information
    optimized_nodes = []
    for node in nodes:
        # Keep all important properties, truncate only very long values
        properties = {}
        for k, v in node.get('properties', {}).items():
            if k.lower() in [prop.lower() for prop in important_properties]:
                # Keep important properties, but limit very long text
                if isinstance(v, str) and len(v) > 200:
                    properties[k] = v[:200] + '...'
                else:
                    properties[k] = v
            elif len(str(v)) <= 50:  # Keep short properties regardless
                properties[k] = v
        
        optimized_nodes.append({
            'id': node['id'],
            'labels': node['labels'],  # Keep all labels for proper categorization
            'properties': properties
        })
    
    # Keep relationship properties but limit size
    optimized_relationships = []
    for rel in relationships:
        rel_properties = {}
        for k, v in rel.get('properties', {}).items():
            if len(str(v)) <= 100:  # Keep reasonably sized relationship properties
                rel_properties[k] = v
        
        optimized_relationships.append({
            'source': rel['source'],
            'target': rel['target'],
            'type': rel['type'],
            'properties': rel_properties
        })
    
    graph_data = {
        'nodes': optimized_nodes,
        'relationships': optimized_relationships,
        'stats': {
            'nodes': len(optimized_nodes),
            'relationships': len(optimized_relationships)
        }
    }
    
    # Try to compress and store with larger limit for medical data
    compressed = compress_data(graph_data)
    if compressed and len(compressed) < 3500:  # Increased limit for medical properties
        session['graph_data_compressed'] = compressed
        session['has_graph_data'] = True
        print(f"Graph data compressed to {len(compressed)} bytes and stored in session")
        return True
    else:
        # If still too large, try with more aggressive optimization
        print(f"Graph data too large ({len(compressed) if compressed else 'N/A'} bytes), trying aggressive optimization...")
        
        # More aggressive optimization - keep only most essential properties
        essential_props = ['name', 'umls_cui', 'umls_preferred_name', 'type']
        minimal_nodes = []
        for node in nodes:
            properties = {}
            for k, v in node.get('properties', {}).items():
                if k.lower() in [prop.lower() for prop in essential_props]:
                    if isinstance(v, str) and len(v) > 100:
                        properties[k] = v[:100] + '...'
                    else:
                        properties[k] = v
            
            minimal_nodes.append({
                'id': node['id'],
                'labels': node['labels'][:1],  # Keep only first label
                'properties': properties
            })
        
        minimal_data = {
            'nodes': minimal_nodes,
            'relationships': [{'source': r['source'], 'target': r['target'], 'type': r['type']} for r in optimized_relationships],
            'stats': graph_data['stats']
        }
        
        compressed_minimal = compress_data(minimal_data)
        if compressed_minimal and len(compressed_minimal) < 3500:
            session['graph_data_compressed'] = compressed_minimal
            session['has_graph_data'] = True
            print(f"Minimal graph data compressed to {len(compressed_minimal)} bytes and stored in session")
            return True
        else:
            # Last resort - store only statistics
            session['graph_stats'] = graph_data['stats']
            session['has_graph_data'] = False
            print("Graph data too large even with aggressive optimization, storing only statistics")
            return False

def get_graph_data():
    """Retrieve graph data from session."""
    if session.get('has_graph_data'):
        compressed = session.get('graph_data_compressed')
        return decompress_data(compressed)
    else:
        # Return minimal data if full graph not available
        return {
            'nodes': [],
            'relationships': [],
            'stats': session.get('graph_stats', {'nodes': 0, 'relationships': 0})
        }

from queue import Queue
from threading import Lock

# Thread-safe queue and status management
class ProcessingManager:
    def __init__(self):
        self.message_queue = Queue()
        self.status_lock = Lock()
        self.complete = False
        self.success = False
        self.error = None
        self.result = None
        self._messages = []
    
    def reset(self):
        with self.status_lock:
            while not self.message_queue.empty():
                self.message_queue.get()
            self._messages = []
            self.complete = False
            self.success = False
            self.error = None
            self.result = None
    
    def add_message(self, message):
        self.message_queue.put(message)
        with self.status_lock:
            self._messages.append(message)
    
    def get_status(self):
        with self.status_lock:
            return {
                'messages': self._messages.copy(),
                'complete': self.complete,
                'success': self.success,
                'error': self.error
            }
    
    def set_complete(self, success, error=None, result=None):
        with self.status_lock:
            self.complete = True
            self.success = success
            self.error = error
            self.result = result

# Global processing manager
processing_manager = ProcessingManager()

def process_pdf(filepath, filename):
    try:
        # Reset processing status
        processing_manager.reset()

        # Step 1: OCR
        processing_manager.add_message(f'⚙️ Extracting text from "{filename}"...')
        ocr_text = ocr_processor.extract_text(filepath)
        processing_manager.add_message(f'✅ Document "{filename}" text extracted successfully!')

        # Step 2: Semantic
        processing_manager.add_message('🔄 Running semantic analysis...')
        semantic_output = semantic_processor.process(ocr_text)
        processing_manager.add_message('✅ Semantic analysis completed successfully!')
        
        # Create temporary file for Pinata upload
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write(semantic_output)
            semantic_file = temp_file.name

        # Step 3: Pinata
        processing_manager.add_message('☁️ Uploading document to IPFS network...')
        pinata_link = pinata_uploader.upload_document(semantic_file)
        processing_manager.add_message('✅ Document successfully stored on IPFS!')
        
        # Clean up temporary file
        os.unlink(semantic_file)

        # Step 4: NER
        processing_manager.add_message('🔄 Extracting medical entities and relationships...')
        graph_elements = ner_agent.extract_from_text(semantic_output)
        if not graph_elements:
            raise Exception("No medical entities could be extracted from the document.")
        
        if len(graph_elements) > 1:
            processing_manager.add_message('🔄 Merging extracted knowledge elements...')
            merged = ner_agent.merge_graph_elements(graph_elements)
        else:
            merged = graph_elements[0]

        if validate_graph_element(merged):
            processing_manager.add_message('🔄 Building knowledge graph structure...')
            # Convert graph elements to session-friendly format
            nodes = []
            relationships = []
            
            for node in merged.nodes:
                nodes.append({
                    'id': node.id,
                    'labels': [node.type],
                    'properties': node.properties
                })
            
            for rel in merged.relationships:
                relationships.append({
                    'source': rel.subj.id,
                    'target': rel.obj.id,
                    'type': rel.type,
                    'properties': rel.properties
                })
            
            processing_manager.add_message(f'✅ Knowledge graph created with {len(nodes)} entities and {len(relationships)} relationships!')
            
            result = {
                'pinata_link': pinata_link,
                'nodes': nodes,
                'relationships': relationships,
                'filename': os.path.basename(filepath)
            }
            
            processing_manager.set_complete(True, result=result)
            processing_manager.add_message('✅ Document successfully processed!')
            
            return {'success': True}
        else:
            raise Exception("Invalid graph element structure.")
            
    except Exception as e:
        error_msg = str(e)
        print(f"Error in process_pdf: {error_msg}")
        processing_manager.set_complete(False, error=error_msg)
        return {'success': False, 'error': error_msg}

@app.route('/knowledge_graph')
def knowledge_graph():
    return render_template('knowledge_graph.html')

@app.route('/get_graph_data')
def get_graph_data_route():
    try:
        # Get graph data from session using our helper function
        graph_data = get_graph_data()
        
        # Check if we have graph data
        if not graph_data or (not graph_data.get('nodes') and not graph_data.get('relationships')):
            return jsonify({
                'success': False,
                'error': 'No graph data available. Please upload and process a document first.'
            })
        
        return jsonify({
            'success': True,
            'stats': graph_data['stats'],
            'nodes': graph_data['nodes'],
            'relationships': graph_data['relationships']
        })
    except Exception as e:
        print(f"Error in get_graph_data: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/check_progress')
def check_progress():
    status = processing_manager.get_status()
    
    # If processing is complete and successful, update session with results
    if status['complete'] and status['success'] and processing_manager.result:
        result = processing_manager.result
        session['pinata_link'] = result['pinata_link']
        session['pdf_file'] = result['filename']
        session['processing_complete'] = True
        
        # Store graph data in session
        store_graph_data(result['nodes'], result['relationships'])
    
    return jsonify(status)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            # Handle AJAX request
            if 'file' not in request.files:
                return jsonify({'success': False, 'error': 'No file selected'})
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No file selected'})
            
            if not allowed_file(file.filename):
                return jsonify({'success': False, 'error': 'Please upload only PDF files'})
            
            try:
                filename = file.filename
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Clear previous session data and processing status
                session.clear()
                processing_manager.reset()
                
                # Add initial upload success message
                processing_manager.add_message(f'📄 Document "{filename}" uploaded successfully!')
                
                # Process in a separate thread to not block
                from threading import Thread
                thread = Thread(target=process_pdf, args=(filepath, filename))
                thread.daemon = True  # Ensure thread is terminated when main thread ends
                thread.start()
                
                return jsonify({
                    'success': True,
                    'message': f'Started processing {filename}'
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                })
        else:
            # Handle regular form submission
            if 'file' not in request.files:
                flash('⚠️ No file selected')
                return redirect(request.url)
            
            file = request.files['file']
            if file.filename == '':
                flash('⚠️ No file selected')
                return redirect(request.url)
            
            if not allowed_file(file.filename):
                flash('⚠️ Please upload only PDF files')
                return redirect(request.url)
            
            try:
                filename = file.filename
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Clear previous session data before processing new file
                session.clear()
                
                # Show initial upload success message
                flash(f'📄 Document "{filename}" uploaded successfully!')
                
                # Start processing in background
                process_pdf(filepath, filename)
                
                return redirect(url_for('upload_file'))
            except Exception as e:
                flash('❌ Error: ' + str(e))
                return redirect(request.url)
    
    # For GET requests, always start with a clean state
    # Only show data if there's a current session with processed file
    current_pdf = session.get('pdf_file') if session.get('processing_complete') else None
    current_pinata = session.get('pinata_link') if session.get('processing_complete') else None
    
    return render_template('upload.html',
                         pdf_file=current_pdf,
                         pinata_link=current_pinata)

@app.route('/clear_session')
def clear_session():
    """Clear all session data and redirect to home"""
    session.clear()
    flash('Session cleared. You can now upload a new document.')
    return redirect(url_for('upload_file'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, mimetype='application/pdf')

@app.route('/pdf/<filename>')
def serve_pdf(filename):
    """Serve PDF file directly for iframe viewing"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, mimetype='application/pdf')

@app.route('/view/<filename>')
def view_pdf(filename):
    try:
        # Get data from session
        pinata_link = session.get('pinata_link', '')
        graph_data = get_graph_data()
        
        # Transform nodes for visualization
        transformed_nodes = []
        for node in graph_data.get('nodes', []):
            transformed_nodes.append({
                'id': node['id'],
                'labels': node['labels'],
                'properties': node.get('properties', {}),
                'title': node.get('properties', {}).get('name', node['id']).replace('_', ' ').title()
            })
        
        # Transform relationships for visualization
        links = []
        for rel in graph_data.get('relationships', []):
            links.append({
                'source': rel['source'],
                'target': rel['target'],
                'type': rel['type'],
                'properties': rel.get('properties', {})
            })
        
        # Prepare the complete graph data
        complete_graph_data = {
            'nodes': transformed_nodes,
            'links': links,
            'stats': {
                'totalNodes': graph_data['stats']['nodes'],
                'totalRelationships': graph_data['stats']['relationships']
            }
        }
        
        return render_template('view_pdf.html',
                             filename=filename,
                             graph_data=json.dumps(complete_graph_data),
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
    port = int(os.getenv('PORT', 5005))  # Use Render's PORT or default to 5000 locally
    app.run(host='0.0.0.0', port=port, debug=False)

