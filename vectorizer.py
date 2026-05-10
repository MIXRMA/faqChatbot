import math
from collections import defaultdict


class TfidfVectorizerCustom:
  
    def __init__(self):
        self.vocabulary: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.docCount: int = 0

    def fit(self, corpus: list[str]) -> None:
    
        self.docCount = len(corpus)
        dfCounts: dict[str, int] = defaultdict(int)

        for doc in corpus:
            uniqueWords = set(doc.split())
            for word in uniqueWords:
                dfCounts[word] += 1

        for idx, word in enumerate(sorted(dfCounts.keys())):
            self.vocabulary[word] = idx
            self.idf[word] = math.log(
                (self.docCount + 1) / (dfCounts[word] + 1)
            ) + 1

    def _vectorize(self, doc: str) -> list[float]:
       
        vocabSize = len(self.vocabulary)
        vector = [0.0] * vocabSize
        tokens = doc.split()

        if not tokens:
            return vector

        tfCounts: dict[str, int] = defaultdict(int)
        for token in tokens:
            tfCounts[token] += 1

        for word, count in tfCounts.items():
            if word in self.vocabulary:
                tf = count / len(tokens)
                vector[self.vocabulary[word]] = tf * self.idf[word]

        return vector

    def transform(self, corpus: list[str]) -> list[list[float]]:
        return [self._vectorize(doc) for doc in corpus]

    def fitTransform(self, corpus: list[str]) -> list[list[float]]:
        self.fit(corpus)
        return self.transform(corpus)

    def transformQuery(self, query: str) -> list[float]:
        
        return self._vectorize(query)