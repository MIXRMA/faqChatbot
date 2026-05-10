import json
from src.preprocessor import preprocessToString
from src.vectorizer import TfidfVectorizerCustom
from src.matcher import findBestMatch

FALLBACK_THRESHOLD = 0.10
LOW_CONF_THRESHOLD = 0.20

FALLBACK_RESPONSE = (
    "I'm sorry, I couldn't find a relevant answer in my knowledge base for that query. "
    "Please try rephrasing, or contact the university helpdesk directly for assistance."
)

LOW_CONFIDENCE_PREFIX = (
    "I'm not entirely certain, but the closest answer I have is:\n\n"
)


class FaqChatbot:
   
    def __init__(self, faqPath: str):
        with open(faqPath, 'r', encoding='utf-8') as f:
            self.faqs: list[dict] = json.load(f)

        self.questions: list[str] = [item['question'] for item in self.faqs]
        self.answers: list[str] = [item['answer'] for item in self.faqs]

        # Preprocess all FAQ questions once at startup
        self.processedQuestions: list[str] = [
            preprocessToString(q) for q in self.questions
        ]

        # Fit TF-IDF on the FAQ corpus and cache all FAQ vectors
        self.vectorizer = TfidfVectorizerCustom()
        self.faqVectors: list[list[float]] = self.vectorizer.fitTransform(
            self.processedQuestions
        )

    def respond(self, userInput: str) -> dict:
        """
        Process a raw user query and return a response dict with keys:
            - answer  (str)  : the text response to show the user
            - score   (float): cosine similarity of the best match
            - matched (str)  : the FAQ question that was matched (or empty)
            - status  (str)  : 'ok' | 'low_confidence' | 'fallback' | 'empty'

        Returning a dict instead of a plain string makes it easy for the
        Flask layer to pass metadata (confidence, matched question) to the
        frontend for display.
        """
        processedInput = preprocessToString(userInput)

        if not processedInput.strip():
            return {
                'answer': "Please type a question and I'll do my best to help.",
                'score': 0.0,
                'matched': '',
                'status': 'empty'
            }

        queryVector = self.vectorizer.transformQuery(processedInput)
        bestIdx, bestScore = findBestMatch(queryVector, self.faqVectors)

        if bestScore < FALLBACK_THRESHOLD:
            return {
                'answer': FALLBACK_RESPONSE,
                'score': round(bestScore, 4),
                'matched': '',
                'status': 'fallback'
            }

        if bestScore < LOW_CONF_THRESHOLD:
            return {
                'answer': LOW_CONFIDENCE_PREFIX + self.answers[bestIdx],
                'score': round(bestScore, 4),
                'matched': self.questions[bestIdx],
                'status': 'low_confidence'
            }

        return {
            'answer': self.answers[bestIdx],
            'score': round(bestScore, 4),
            'matched': self.questions[bestIdx],
            'status': 'ok'
        }