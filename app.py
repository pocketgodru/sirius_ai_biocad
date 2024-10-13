from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import datetime
from gradio_client import Client
from llm import init
from file_to_text import load_text

# Flask app
app = Flask(__name__)

# Folder to save uploaded files
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf' , 'txt' , 'csv' , 'doc' , 'docx'}  # Указываем допустимые форматы файлов
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize the Gradio Client for the Hugging Face Space
client = Client("Qwen/Qwen2.5")

# Dictionary to store uploaded text for session
uploaded_text = {}

# Check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_file():
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        if file and allowed_file(file.filename):  # Проверяем формат загружаемого файла
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Extract text from PDF file
            text = load_text(filepath)
        else:
            return jsonify({"result": "Неподдерживаемый формат файла. Пожалуйста, загрузите PDF файл."})
    elif 'text' in request.form and request.form['text'].strip() != '':
        text = request.form['text']
    else:
        return jsonify({"result": "No file or text provided"})

    # Save text to uploaded_text dictionary with a session-based key
    session_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    uploaded_text[session_id] = text

    return jsonify({"session_id": session_id, "result": "Text uploaded successfully."})

@app.route('/summarize', methods=['POST'])
def summarize_text():
    session_id = request.form.get('session_id')
    if session_id in uploaded_text:
        text = uploaded_text[session_id]
        # Perform summarization on text using Hugging Face Space
        summarized_text = init(client, text)
        return jsonify({"result": f"{summarized_text}"})
    else:
        return jsonify({"result": "No uploaded text found."})

@app.route('/question', methods=['POST'])
def ask_question():
    session_id = request.form.get('session_id')
    question = request.form.get('question')
    if session_id in uploaded_text:
        text = uploaded_text[session_id]
        # Perform question answering on text using Hugging Face Space
        answer = init(client, text, question)
        return jsonify({"result": f"{answer}"})
    else:
        return jsonify({"result": "No uploaded text found."})

if __name__ == '__main__':
    app.run()