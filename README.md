
# Email Classifier


A web application that automatically classifies emails as productive (requiring action) or unproductive (irrelevant or spam) and generates context-aware responses. The application supports TXT and PDF uploads, uses Naive Bayes as a fallback, and integrates OpenAI GPT for AI-powered responses.
---

## Project Structure

```
AUTOU_CHALLENGE/
│
├─ app.py               # Main assistant code
├─ local_email_model.pkl            # Saved Naive Bayes model (fallback)
├─ local_vectorizer.pkl             # Saved vectorizer for text processing
├─ requirements.txt       # Project dependencies
├─ README.md              
└─ templates/                   
   ├─ index.html     
└─ static/                   
   ├─ style.css               
└─ classifiers/                   # MCP protocol module
   ├─ email_classifier.py   # Naive Bayes fallback and OpenAI classification
   ├─ preprocess.py          # Text cleaning and keyword extraction
   ├─ response_generator.py  # Generates AI or fallback responses
```
---

## Installation & Setup

1. **Clone this repository:**

```bash
git clone https://github.com/gabriellemosc/AutoU_Challenge.git
cd AutoU_Challenge
````

2. **Create and activate a virtual environment (optional but recommended):**

```bash
python3 -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

3. **Install the dependencies**

```
pip install -r requirements.txt
```


3. **Add your OPENAI API KEY**

```
OPENAI_API_KEY=your_openai_api_key_here
```


5. **Start the Flask server in a your terminal:**

```bash
python app.py
```
Open your browser at http://127.0.0.1:5000
---

## How to Use


1. **Upload or paste an email:**
    OBS: Maximum upload size: 16MB and Supported formats: .txt and .pdf
   
   * Send an email in text format for analysis or send a PDF file
   * The agent will give the answer as productive or unproductive.
   * You can paste the suggested answer given by the analyst.

2. **Classification & Response:**

    The system will automatically classify the email as:
    
    Produtivo (Productive): Requires action such as a task, meeting, client request, or professional communication.
    
    Improdutivo (Unproductive): Spam, promotions, marketing, irrelevant messages, prizes, chains, or emails not requiring action.
    
    Generates a suggested response based on the classification.

---

## Technical Overview

  Algorithm:
  
  Emails are first preprocessed (text normalization, cleaning, stopwords removal, keyword extraction).
  
  Classification: The system attempts to use OpenAI GPT to classify the email.
  
  Fallback: If GPT fails, a Naive Bayes model with a TF-IDF vectorizer is used.

### Technologies:

  Flask: Web framework for serving the application.
  
  PyPDF2: Extracts text from uploaded PDF files.
  
  OpenAI GPT: AI-powered classification and response generation.
  
  scikit-learn: Naive Bayes classifier and vectorizer.



---
## Contact

**Gabriel Lemos**

* GitHub: [https://github.com/gabriellemosc](https://github.com/gabriellemosc)

