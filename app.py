import os
import uuid  # Для генерации уникальных session_id
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from utils.file_to_text import process_file
from utils.llm import *
import utils.search_article as sa
import utils.lit_obr as lt

application = Flask(__name__)

# Настройки
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'csv', 'docx'}
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

uploaded_text = {}

# Создаем папку для загрузки файлов, если её нет
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(f'../{UPLOAD_FOLDER}')

# Функция для проверки допустимых расширений файлов
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@application.route('/')
def home():
    return render_template('index.html')

# Эндпоинт для загрузки файлов
@application.route('/process', methods=['POST'])
def process_files():
    if 'files' not in request.files:
        return jsonify({"result": "No files part in the request"}), 400

    uploaded_files = request.files.getlist('files')
    if not uploaded_files:
        return jsonify({"result": "No files selected"}), 400

    # Создаем уникальный session_id для пользователя
    session_id = str(uuid.uuid4())

    # Инициализируем данные для этой сессии
    session_data = {"text": "", "images": []}

    for file in uploaded_files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(application.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)  # Сохраняем файл на сервер

            # Извлечение текста и изображений
            text, images = process_file(filepath)

            if text is not None:
                session_data["text"] += f"\n\nСтатья название ниже ----------\n" + text  # Добавляем текст в общую строку
                session_data["images"].extend(images)  # Добавляем изображения
            else:
                return jsonify({"result": f"Unsupported file type: {filename}"}), 400

    # Сохраняем данные в глобальный объект uploaded_text
    uploaded_text[session_id] = session_data

    return jsonify({
        "result": "Files processed successfully",
        "session_id": session_id,
        "text_preview": session_data["text"]
    })

@application.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    session_id = data.get('session_id')
    percentage = data.get('percentage')  # Процент сжатия
    preferences = data.get('preferences')  
    if session_id in uploaded_text:
        
        text = uploaded_text[session_id]["text"]
        images = uploaded_text[session_id]["images"]

        # Передаем данные в функцию суммаризации
        summarized_text  = init_summ(text, preferences, images, percentage)
        return jsonify({
            "result": summarized_text
        })
    else:
        return jsonify({"result": "No uploaded text found."}), 400


@application.route('/question', methods=['POST'])
def ask_question():
    data = request.json
    session_id = data.get('session_id')
    question = data.get('question')

    if session_id in uploaded_text:
        text = uploaded_text[session_id]["text"]
        images = uploaded_text[session_id]["images"]

        # Вопрос-ответ
        answer= init_qa(text, question, images)
        return jsonify({
            "result": answer,
        })
    else:
        return jsonify({"result": "No uploaded text found."}), 400


@application.route('/article-recommendation', methods=['POST'])
def article_recommendation():
    data = request.json
    session_id = data.get('session_id')
    if session_id in uploaded_text:
        text = uploaded_text[session_id]["text"]
        images = uploaded_text[session_id]["images"]

        # Рекомендация статей
        topics, result_article_json = sa.init(text , images)
        return jsonify({
            "topics": topics,
            "articles": result_article_json  # Return articles with topics
        })
    else:
        return jsonify({"result": "No uploaded text found."}), 400

@application.route('/literature-review', methods=['POST'])
def l_r():
    data = request.json
    session_id = data.get('session_id')
    criteria = data.get('criteria')

    if session_id in uploaded_text:
        text = uploaded_text[session_id]["text"]
        images = uploaded_text[session_id]["images"]
        
        ret = lt.init(text , criteria)
        return ret
    else:
        return {"result": "No uploaded text found."}, 400


if __name__ == '__main__':
    application.run(host='0.0.0.0')




'''import os
import uuid  # Для генерации уникальных session_id
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from utils.file_to_text import process_file
from utils.new_bek import *

application = Flask(__name__)

# Настройки
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'csv', 'docx'}
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

uploaded_text = {}

# Создаем папку для загрузки файлов, если её нет
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Функция для проверки допустимых расширений файлов
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@application.route('/')
def home():
    return render_template('new_index.html')

# Эндпоинт для загрузки файлов
@application.route('/process', methods=['POST'])
def process_files():
    if 'files' not in request.files:
        return jsonify({"result": "No files part in the request"}), 400

    uploaded_files = request.files.getlist('files')
    if not uploaded_files:
        return jsonify({"result": "No files selected"}), 400

    # Создаем уникальный session_id для пользователя
    session_id = str(uuid.uuid4())

    # Инициализируем данные для этой сессии
    session_data = {"text": "", "images": []}

    for file in uploaded_files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(application.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)  # Сохраняем файл на сервер

            # Извлечение текста и изображений
            text, images = process_file(filepath)

            if text is not None:
                session_data["text"] += f"\n\nСтатья название ниже ----------\n" + text  # Добавляем текст в общую строку
                session_data["images"].extend(images)  # Добавляем изображения
            else:
                return jsonify({"result": f"Unsupported file type: {filename}"}), 400

    # Сохраняем данные в глобальный объект uploaded_text
    uploaded_text[session_id] = session_data

    return jsonify({
        "result": "Files processed successfully",
        "session_id": session_id,
        "text_preview": session_data["text"]
    })

@application.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    session_id = data.get('session_id')
    percentage = data.get('percentage')  # Процент сжатия
    preferences = data.get('preferences')  
    if session_id in uploaded_text:
        
        text = uploaded_text[session_id]["text"]
        images = uploaded_text[session_id]["images"]

        # Передаем данные в функцию суммаризации
        summarized_text  = summarize_text(text, preferences, images, percentage)
        return jsonify({
            "result": summarized_text['Summary']
        })
    else:
        return jsonify({"result": "No uploaded text found."}), 400


@application.route('/question', methods=['POST'])
def ask_question():
    data = request.json
    session_id = data.get('session_id')
    question = data.get('question')

    if session_id in uploaded_text:
        text = uploaded_text[session_id]["text"]
        images = uploaded_text[session_id]["images"]

        # Вопрос-ответ
        answer= answer_question(text, question, images)
        return jsonify({
            "result": answer['Answer'],
        })
    else:
        return jsonify({"result": "No uploaded text found."}), 400


@application.route('/article-recommendation', methods=['POST'])
def article_recommendation():
    data = request.json
    session_id = data.get('session_id')
    if session_id in uploaded_text:
        text = uploaded_text[session_id]["text"]
        images = uploaded_text[session_id]["images"]

        # Рекомендация статей
        topics, result_article_json = search_article(text)
        return jsonify({
            "topics": topics,
            "articles": result_article_json  # Return articles with topics
        })
    else:
        return jsonify({"result": "No uploaded text found."}), 400

@application.route('/literature-review', methods=['POST'])
def l_r():
    data = request.json
    session_id = data.get('session_id')
    criteria = data.get('criteria')

    if session_id in uploaded_text:
        text = uploaded_text[session_id]["text"]
        images = uploaded_text[session_id]["images"]
        
        ret = literature_review(text , criteria)
        return ret
    else:
        return {"result": "No uploaded text found."}, 400


if __name__ == '__main__':
    application.run(host='0.0.0.0')
'''