"""Cajueira: componentes didáticos e extensíveis de NLP e IA."""

from cajueira.nlp import Vocabulary, normalize_text, tokenize
from cajueira.vectors import CountVectorizer, TfidfVectorizer
from cajueira.math import cosine_similarity, dot_product, vector_norm
from cajueira.learning import MultinomialNaiveBayes
from cajueira.neural import MLPClassifier
from cajueira.routing import IntentRouter, IntentResult

__version__ = "0.1.1"

__all__ = [
    "Vocabulary",
    "normalize_text",
    "tokenize",
    "CountVectorizer",
    "TfidfVectorizer",
    "cosine_similarity",
    "dot_product",
    "vector_norm",
    "MultinomialNaiveBayes",
    "MLPClassifier",
    "IntentRouter",
    "IntentResult",
]
