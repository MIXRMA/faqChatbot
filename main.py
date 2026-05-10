import nltk

def ensureNltkData():
    required = [
        ('tokenizers/punkt',            'punkt'),
        ('tokenizers/punkt_tab',        'punkt_tab'),
        ('corpora/stopwords',           'stopwords'),
        ('corpora/wordnet',             'wordnet'),
    ]
    for path, name in required:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"[setup] Downloading NLTK resource: {name}")
            nltk.download(name, quiet=True)

ensureNltkData()

from src.chatbot import FaqChatbot  # noqa: E402 — import after NLTK setup

STATUS_LABELS = {
    'ok':             '✓',
    'low_confidence': '⚠',
    'fallback':       '✗',
    'empty':          '–',
}


def main():
    print("\n" + "═" * 52)
    print("  University FAQ Chatbot  |  NLTK + TF-IDF")
    print("  Type 'quit' or 'exit' to stop.")
    print("═" * 52 + "\n")

    bot = FaqChatbot(faqPath='data/faqs.json')
    print(f"  Loaded {len(bot.faqs)} FAQ entries. Ready.\n")

    while True:
        try:
            userInput = input("You  › ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot  › Goodbye!")
            break

        if not userInput:
            continue

        if userInput.lower() in ('quit', 'exit', 'bye', 'q'):
            print("Bot  › Goodbye!")
            break

        result = bot.respond(userInput)
        icon   = STATUS_LABELS.get(result['status'], '?')
        score  = f"{result['score'] * 100:.1f}%"

        print(f"\nBot  › {result['answer']}")
        print(f"       {icon} {result['status'].upper()}  |  score: {score}", end='')
        if result['matched']:
            print(f"  |  matched: {result['matched']}", end='')
        print("\n")


if __name__ == '__main__':
    main()