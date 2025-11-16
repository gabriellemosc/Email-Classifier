from flask import Flask, request, jsonify, render_template
import os
#import PyPDF2
#from classifiers.email_classifier import classify_email
#from classifiers.response_generator import generate_response


app = Flask(__name__)

@app.route('/')
def home():
    return "Running"

if __name__ == '__main__':
    app.run(debug=True)