"""Regression cases for reviewed multilingual meeting input."""

from regional_translation import RegionalTranslator


CASES = (
    ("en-IN", "hello", "hello"),
    ("hi-IN", "नमस्ते", "hello"),
    ("hi-IN", "मुझे समझ नहीं आया", "i do not understand"),
    ("hi-IN", "क्या आप मुझे सुन सकते हैं", "can you hear me"),
    ("ml-IN", "നമസ്കാരം", "hello"),
    ("ml-IN", "എനിക്ക് മനസ്സിലായില്ല", "i do not understand"),
    ("ml-IN", "ദയവായി പതുക്കെ പറയൂ", "please speak slowly"),
    ("ta-IN", "வணக்கம்", "hello"),
    ("ta-IN", "எனக்கு புரியவில்லை", "i do not understand"),
    ("ta-IN", "மீண்டும் சொல்லுங்கள்", "repeat that"),
)


def main():
    translator = RegionalTranslator()
    for language, source, expected in CASES:
        result = translator.translate(source, language)
        assert result["english"] == expected, (language, source, result)
    print(f"multilingual reviewed regression passed: {len(CASES)} cases")


if __name__ == "__main__":
    main()
