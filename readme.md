# Introduction: FAQ Chatbot Using NLTK
This is a retrieval-based FAQ chatbot that makes use of the NLTK library. It operates on a fixed JSON dataset of question-answer pairs. At runtime, an input query by the user is preprocessed through an NLP pipeline (Tokenization -> Stopward Removal -> Lemmatization), and then vectorized using a manually implemented TF-IDF engine. This is not the built-in vectorizer from ```scikit-learn``` but a from-scratch implementation using smooth IDF, which is expressed mathematically by

$\log\left(\frac{N+1}{df+1}\right) + 1$

Cosine similarity is computed between the query vector and all FAQ question vectors, and a three-tier confidence gating mechanism (With thresholds at 0.10 and 0.20) determine whether to return a matching answer, low-confidence warning or a fallback response. This chatbot is served via a Flask web interface on Port 8080 (The standard is to host this on Port 5000, but that is already used by the AirPlay/AirPort Utility processes on macOS).

# Python Libraries
1. ```nltk```: This library is the backbone of the entire NLP preprocessing pipeline, providing tokenization (```punkt```), stopward lists and the Word lemmatizer used to normalize use queries and FAQ questions before vectorization.
2. ```scikit-learn```: This library supplies the ```TfidfVectorizer``` and ```cosine_similarity``` utilities used as reference or fallback implementations alongside the engine built in ```vectorizer.py```.
3. ```Flask```: This library hosts the web interface on Port 8080 that accepts user queries via a text input form and then returns matched FAQ answers through the interface pipeline of the chatbot.

# Training Model
In this project, no ML model is used in the traditional sense of the term since the entire pipeline is retrieval-based. Rather, the core engine of this entire project is the aforementioned from-scratch TF-IDF vectorizer with cosine similarity matching. As a result, there is no training phase as the chatbot is ready immediately on launch once the NLTK corpora, i.e. ```punkt```, ```punkt_tab```, ```stopwords``` and ```wordnet```, are downloaded.

# Project Directory (Textual Format)
<img width="237" height="356" alt="image" src="https://github.com/user-attachments/assets/dc88080c-133e-460e-909a-317695692682"/>

# Running Instructions
1. Organise each and every file in this repo according to the screenshot of the project directory attached above in exactly that order.
2. Set up the Python virtual environment (```venv```) in the Terminal and install the libraries by running the command ```pip install -r requirements.txt```.
3. Run the command ```python app.py```. The Terminal will return ```http://127.0.0.1:8080```; Copy paste that link in any web browser to open the chatbot.

# Output Example
<img width="728" height="770" alt="image" src="https://github.com/user-attachments/assets/f5f211bc-710e-4e8e-b8c8-150812f7c515"/>
