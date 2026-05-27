import shutil
import re

import cv2


class OCRReader:
    def __init__(self):
        self.available = shutil.which("tesseract") is not None
        self._pytesseract = None
        self.languages = []
        if self.available:
            try:
                import pytesseract

                self._pytesseract = pytesseract
                self.languages = self._available_languages()
            except ImportError:
                self.available = False

    def read(self, image) -> str:
        if not self.available or self._pytesseract is None:
            return ""

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        data = self._pytesseract.image_to_data(
            gray,
            lang=self.language_code,
            config="--psm 6",
            output_type=self._pytesseract.Output.DICT,
        )
        words = []
        scored_words = []
        for word, confidence in zip(data.get("text", []), data.get("conf", [])):
            token = word.strip()
            score = self._confidence(confidence)
            if self._keep_word(word, confidence):
                words.append(token)
                scored_words.append((token, score))

        text = " ".join(words[:30])
        return text if len(text) >= 3 and self._looks_like_real_text(scored_words) else ""

    @property
    def language_code(self):
        preferred = [lang for lang in ("eng", "nep") if lang in self.languages]
        return "+".join(preferred) if preferred else "eng"

    def _available_languages(self):
        try:
            return self._pytesseract.get_languages(config="")
        except Exception:
            return []

    def _keep_word(self, word, confidence):
        token = word.strip()
        if not token:
            return False
        score = self._confidence(confidence)
        if score is None:
            return False
        return len(token) >= 2 and score >= 55 and re.search(r"[A-Za-z0-9\u0900-\u097F]", token)

    def _confidence(self, confidence):
        try:
            return float(confidence)
        except ValueError:
            return None

    def _looks_like_real_text(self, scored_words):
        alpha_words = [
            (word, score)
            for word, score in scored_words
            if score is not None and re.search(r"[A-Za-z\u0900-\u097F]", word)
        ]
        if not alpha_words:
            return False

        avg_confidence = sum(score for _, score in alpha_words) / len(alpha_words)
        has_long_word = any(len(word) >= 3 for word, _ in alpha_words)
        return avg_confidence >= 70 and (has_long_word or len(alpha_words) >= 3)
