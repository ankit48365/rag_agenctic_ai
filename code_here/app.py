# >> calls >> config.py >> model.py >> app.py

from flask import Flask, request, jsonify, render_template
from model import opus_response, sonet_response, haiku_response, json_parser
import time
import os

# Configure Flask to look for templates in the parent directory's templates folder
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
app = Flask(__name__, template_folder=template_dir)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    user_message = data.get('message')
    model = data.get('model')
    
    if not user_message or not model:
        return jsonify({"error": "Missing message or model selection"}), 400
    
    system_prompt = f"""You are an AI assistant helping with customer inquiries. 
    You must respond with valid JSON only, no additional text or explanations.
    
    {json_parser.get_format_instructions()}
    
    Analyze the user's message and provide the required JSON response."""
    
    start_time = time.time()
    

    try:
        if model == 'opus_ll':
            result = opus_response(system_prompt, user_message)
        elif model == 'haiku_ll':
            result = haiku_response(system_prompt, user_message)
        else:
            return jsonify({"error": "Invalid model selection"}), 400
        
        result['duration'] = time.time() - start_time
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
