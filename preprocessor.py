import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess(text: str) -> list[str]:
   
    text = text.lower()
    tokens = word_tokenize(text)

    cleaned = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in string.punctuation and token not in stop_words
    ]

    return cleaned


def preprocessToString(text: str) -> str:
    
    return " ".join(preprocess(text))