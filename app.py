from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import datetime
from utils.llm import *
from utils.file_to_text import process_file
from utils.search_article import *

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'csv', 'doc', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

uploaded_text = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_files():
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            text, images = process_file(filepath)
            
            if text is None:
                return jsonify({"result": "Error in processing file."})
            
        else:
            return jsonify({"result": "Unsupported file format."})
    elif 'text' in request.form and request.form['text'].strip() != '':
        text = request.form['text']
    else:
        return jsonify({"result": "No file or text provided"})

    session_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    uploaded_text[session_id] = {"text": text, "images": images}

    return jsonify({"session_id": session_id, "result": "Text uploaded successfully."})

@app.route('/summarize', methods=['POST'])
def summarize_text():
    session_id = request.form.get('session_id')
    compression_percentage = request.form.get('compression_percentage')  

    if session_id in uploaded_text:
        text = uploaded_text[session_id]["text"]
        images = uploaded_text[session_id]["images"]

        summarized_text, topics, result_article_json = process_article_for_summary(text, images, compression_percentage)
        return jsonify({
            "result": summarized_text,
            "topics": topics,
            "articles": result_article_json
        })
    else:
        return jsonify({"result": "No uploaded text found."})


@app.route('/question', methods=['POST'])
def ask_question():
    session_id = request.form.get('session_id')
    question = request.form.get('question')
    if session_id in uploaded_text:
        text = uploaded_text[session_id]["text"]
        images = uploaded_text[session_id]["images"]

        answer, topics, result_article_json = ask_question_to_mistral(text, question, images)
        return jsonify({
            "result": answer,
            "topics": topics,
            "articles": result_article_json  
        })
    else:
        return jsonify({"result": "No uploaded text found."})

if __name__ == '__main__':
    app.run()
