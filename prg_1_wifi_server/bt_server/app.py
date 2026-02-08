import os
import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024  # Limit upload size to 128MB

# In-memory storage for chat history (resets when server restarts)
chat_history = []

@app.route('/')
def index():
    return render_template('index.html')

# API: Receive text messages
@app.route('/send_msg', methods=['POST'])
def send_msg():
    data = request.json
    content = data.get('message')
    user = data.get('username', 'Anonymous')
    
    if content:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        chat_history.append({
            'time': timestamp,
            'user': user,
            'type': 'text',
            'content': content
        })
    return jsonify({'status': 'success'})

# API: Get all messages
@app.route('/get_msgs')
def get_msgs():
    return jsonify(chat_history)

# API: Handle file uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    user = request.form.get('username', 'Anonymous') # Get username from form data
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        
        # Add file upload event to chat history
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        # Determine if it is an image or a generic file
        msg_type = 'image' if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')) else 'file'
        
        chat_history.append({
            'time': timestamp,
            'user': user,
            'type': msg_type,
            'content': filename,
            'url': f'/uploads/{filename}'
        })
        return jsonify({'status': 'success'})
    
    return jsonify({'status': 'fail'})

# API: Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # host='0.0.0.0' makes the server accessible externally via Bluetooth IP
    app.run(host='0.0.0.0', port=5000, debug=True)