from flask import Flask, request, render_template
import os
from werkzeug.exceptions import RequestEntityTooLarge       #errors: FILE TOO LARGE,
import logging  
from werkzeug.utils import secure_filename      #security against malicious files by user
import PyPDF2                                      #read PDF
from classifiers.response_generator import ResponseGenerator
from classifiers.preprocess import pre_process_text
from classifiers.email_classifier import EmailClassifier


app = Flask(__name__)

# TODO: APP KEY


generator = ResponseGenerator()


#configs
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'txt', 'pdf'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 #16 MB allowed


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS  # check extension 

@app.route('/')
def home():
    return render_template("index.html")    #main html


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return render_template("index.html", error="Arquivo muito grande. Tamanho máximo de 16MB"), 413



@app.route('/process', methods=['POST'])
def process_email():
    content = ""

    email_text = request.form.get("email_text")
    file = request.files.get("file")

    #pasted text
    if email_text and email_text.strip():
        content = email_text.strip()     

    elif file and allowed_file(file.filename):

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        #read text file
        if filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

        #case of PDF
        elif filename.endswith(".pdf"):
            content = ""
            reader = PyPDF2.PdfReader(filepath)
            for page in reader.pages:
                extracted = page.extract_text() or ""
                content += extracted

        #temp file
        os.remove(filepath)

    else:
        return render_template("index.html", error="Você deve dar entrada em um arquivo .txt/.pdf ")
    

    #no content
    if not content.strip():
        return render_template("index.html", category= "Error", response="Sem contexto de email dado")
    
    classification = generator.classifier.classify_with_openai(content)

    if classification is None:
        classification = "Produtivo"

    response = generator.generate_response(content)

    
  

    return render_template("index.html", category=classification, response=response)



