import spacy
import re
nlp = spacy.load('en_core_web_lg')

def preprocessing(text):
    doc = nlp(text)
    filtered_token = [token.text for token in doc if not token.is_stop and not token.is_punct]
    join_text = ' '.join(filtered_token).strip()
    lower_text = join_text.lower()
    return re.sub(r'[^A-Za-z]+',' ',lower_text)



