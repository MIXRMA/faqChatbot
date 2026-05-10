import math

def cosineSimilarity(vecA: list[float], vecB: list[float]) -> float:
    
    dotProduct = sum(a * b for a, b in zip(vecA, vecB))
    magnitudeA = math.sqrt(sum(a ** 2 for a in vecA))
    magnitudeB = math.sqrt(sum(b ** 2 for b in vecB))

    if magnitudeA == 0.0 or magnitudeB == 0.0:
        return 0.0

    return dotProduct / (magnitudeA * magnitudeB)


def findBestMatch(
    queryVector: list[float],
    faqVectors: list[list[float]]
) -> tuple[int, float]:

    bestScore = -1.0
    bestIdx = -1

    for idx, faqVec in enumerate(faqVectors):
        score = cosineSimilarity(queryVector, faqVec)
        if score > bestScore:
            bestScore = score
            bestIdx = idx

    return bestIdx, bestScore