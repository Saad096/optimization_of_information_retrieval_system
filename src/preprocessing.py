import re
from typing import List

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize

# Run once in a Python REPL:
# nltk.download("punkt")
# nltk.download("punkt_tab")
# nltk.download("stopwords")
# nltk.download("wordnet")
# nltk.download("omw-1.4")

import logging
from spellchecker import SpellChecker  # pip install pyspellchecker


class TextPreprocessor:
    def __init__(
        self,
        language: str = "english",
        lowercase: bool = True,
        remove_punctuation: bool = True,
        use_stemming: bool = True,
        use_lemmatization: bool = True,
        fix_spelling: bool = False,      # will be True for queries
        expand_query: bool = False,      # True for queries
        is_query: bool = False,
    ):
        self.language = language
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.fix_spelling = fix_spelling
        self.expand_query = expand_query
        self.is_query = is_query
        self.logger = logging.getLogger("IRSystem")

        self.stop_words = set(stopwords.words(language))
        self.stemmer = nltk.PorterStemmer() if use_stemming else None
        self.lemmatizer = (
            nltk.WordNetLemmatizer() if use_lemmatization else None
        )

        self.punct_pattern = re.compile(r"[^\w\s]")

        self.spell_checker = None
        if fix_spelling:
            spell_lang = self._resolve_spell_lang(language)
            try:
                self.spell_checker = SpellChecker(language=spell_lang)
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.warning(
                    "SpellChecker dictionary for language '%s' unavailable "
                    "(resolved code '%s'); spelling correction disabled. (%s)",
                    language,
                    spell_lang,
                    exc,
                )

    def _basic_clean(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        if self.lowercase:
            text = text.lower()
        if self.remove_punctuation:
            text = self.punct_pattern.sub(" ", text)
        return text

    def _tokenize(self, text: str) -> List[str]:
        return word_tokenize(text)

    def _correct_spelling(self, tokens: List[str]) -> List[str]:
        if not self.spell_checker:
            return tokens
        corrected = []
        for tok in tokens:
            if tok in self.stop_words:
                corrected.append(tok)
            else:
                corrected.append(self.spell_checker.correction(tok))
        return corrected

    def _normalize_tokens(self, tokens: List[str]) -> List[str]:
        norm_tokens = []
        for tok in tokens:
            if tok in self.stop_words:
                continue
            if self.stemmer is not None:
                tok = self.stemmer.stem(tok)
            if self.lemmatizer is not None:
                tok = self.lemmatizer.lemmatize(tok)
            norm_tokens.append(tok)
        return norm_tokens

    def _expand_tokens_with_synonyms(self, tokens: List[str]) -> List[str]:
        expanded = list(tokens)
        for tok in tokens:
            synsets = wordnet.synsets(tok)
            if not synsets:
                continue
            lemmas = {l.name().replace("_", " ") for l in synsets[0].lemmas()}
            # limit to 2 synonyms per word
            expanded.extend(list(lemmas)[:2])
        return expanded

    def preprocess(self, text: str) -> str:
        text = self._basic_clean(text)
        tokens = self._tokenize(text)

        # spelling + expansion only for queries (small)
        if self.is_query and self.fix_spelling:
            tokens = self._correct_spelling(tokens)

        tokens = self._normalize_tokens(tokens)

        if self.is_query and self.expand_query:
            tokens = self._expand_tokens_with_synonyms(tokens)

        return " ".join(tokens)

    @staticmethod
    def _resolve_spell_lang(language: str) -> str:
        """
        Map common language names to pyspellchecker codes.
        Supported codes: en, es, de, fr, pt, ru.
        """
        lang = (language or "").lower()
        mapping = {
            "english": "en",
            "en": "en",
            "eng": "en",
            "spanish": "es",
            "es": "es",
            "german": "de",
            "de": "de",
            "french": "fr",
            "fr": "fr",
            "portuguese": "pt",
            "pt": "pt",
            "russian": "ru",
            "ru": "ru",
        }
        return mapping.get(lang, lang)
