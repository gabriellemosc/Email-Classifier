import re
import unicodedata


def pre_process_text(text):
    """ Process the text to NLP, remove special caracters, normalizes text"""

    if not text or not isinstance(text, str):
        return ""
    
    text = text.lower()

    text = unicodedata.normalize('NFKD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])

    text = re.sub(r'[^a-zA-Z0-9\s.,!?;:]', ' ', text)   #special characters

     # Remover espaços extras
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_keywords(text):
    """ relevant keywords from the text"""

    processed_text = pre_process_text(text)

    #stop words in portuguese - BR
    stop_words = {
        'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas', 'de', 'do', 'da', 'dos', 'das',
        'em', 'no', 'na', 'nos', 'nas', 'por', 'para', 'com', 'sem', 'sob', 'sobre',
        'é', 'são', 'era', 'eram', 'foi', 'foram', 'ser', 'estar', 'está', 'estão',
        'tem', 'tinha', 'teve', 'ter', 'há', 'houve', 'que', 'qual', 'quais', 'quem',
        'onde', 'quando', 'como', 'porque', 'porquê', 'se', 'mas', 'e', 'ou', 'então'
    }

    #filter relevant words
    words = processed_text.split()
    keywords = [word for word in words if word not in stop_words and len(word) > 2]

    return keywords


if __name__ == "__main__":
    # process text
    test_text = "Bom dia, preciso de ajuda com meu login. O sistema travou"
    print("Texto original:", test_text)
    print("Texto processado:", pre_process_text(test_text))
    print("Palavras-chave:", extract_keywords(test_text))