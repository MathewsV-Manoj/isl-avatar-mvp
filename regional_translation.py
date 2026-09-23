"""Offline Indic-language to English translation for the local ISL demo.

Reviewed meeting phrases take priority.  The neural translators are loaded only
when an unlisted utterance arrives, keeping the initial web page responsive.
"""

from __future__ import annotations

from collections import OrderedDict


LANGUAGE_NAMES = {"hi-IN": "Hindi", "ml-IN": "Malayalam", "ta-IN": "Tamil", "en-IN": "English"}

# These translations are deliberately small and reviewed.  They provide an
# immediate, deterministic result for common meeting commands while MT covers
# broader sentences.
REVIEWED_PHRASES = {
    "hi-IN": {
        "नमस्ते": "hello", "सभी को नमस्ते": "hello everyone", "सुप्रभात": "good morning",
        "धन्यवाद": "thank you", "कृपया मेरी मदद करें": "please help me",
        "कृपया दोहराएं": "please repeat", "धीरे बोलिए": "speak slowly",
        "क्या आप मुझे सुन सकते हैं": "can you hear me", "फिर से बोलिए": "repeat that",
        "कृपया धीरे बोलिए": "please speak slowly", "कक्षा शुरू करें": "start the class",
        "आज की कक्षा": "today class", "मेरा फोन": "my phone", "मैं खुश हूँ": "i am happy",
        "मीटिंग शुरू करें": "start the meeting", "मेरा नाम": "my name",
        "मैं ठीक हूँ": "i am fine", "हाँ": "yes", "नहीं": "no",
        "मुझे समझ नहीं आया": "i do not understand",
    },
    "ml-IN": {
        "നമസ്കാരം": "hello", "എല്ലാവർക്കും നമസ്കാരം": "hello everyone", "സുപ്രഭാതം": "good morning",
        "നന്ദി": "thank you", "ദയവായി എന്നെ സഹായിക്കൂ": "please help me",
        "ദയവായി ആവർത്തിക്കൂ": "please repeat", "പതുക്കെ പറയൂ": "speak slowly",
        "എനിക്ക് കേൾക്കാമോ": "can you hear me", "വീണ്ടും പറയൂ": "repeat that",
        "ദയവായി പതുക്കെ പറയൂ": "please speak slowly", "ക്ലാസ് തുടങ്ങാം": "start the class",
        "ഇന്നത്തെ ക്ലാസ്": "today class", "എന്റെ ഫോൺ": "my phone", "എനിക്ക് സന്തോഷമാണ്": "i am happy",
        "യോഗം തുടങ്ങാം": "start the meeting", "എന്റെ പേര്": "my name",
        "എനിക്ക് സുഖമാണ്": "i am fine", "അതെ": "yes", "ഇല്ല": "no",
        "എനിക്ക് മനസ്സിലായില്ല": "i do not understand",
    },
    "ta-IN": {
        "வணக்கம்": "hello", "அனைவருக்கும் வணக்கம்": "hello everyone", "காலை வணக்கம்": "good morning",
        "நன்றி": "thank you", "தயவுசெய்து எனக்கு உதவுங்கள்": "please help me",
        "தயவுசெய்து மீண்டும் சொல்லுங்கள்": "please repeat", "மெதுவாக பேசுங்கள்": "speak slowly",
        "என்னைக் கேட்க முடியுமா": "can you hear me", "மீண்டும் சொல்லுங்கள்": "repeat that",
        "தயவுசெய்து மெதுவாக பேசுங்கள்": "please speak slowly", "வகுப்பை தொடங்கலாம்": "start the class",
        "இன்றைய வகுப்பு": "today class", "என் தொலைபேசி": "my phone", "நான் மகிழ்ச்சியாக இருக்கிறேன்": "i am happy",
        "கூட்டத்தை தொடங்கலாம்": "start the meeting", "என் பெயர்": "my name",
        "நான் நலமாக இருக்கிறேன்": "i am fine", "ஆம்": "yes", "இல்லை": "no",
        "எனக்கு புரியவில்லை": "i do not understand",
    },
}

# The Hindi OPUS Marian model is used for broad Hindi input. Tamil and Malayalam
# remain reviewed-phrase-only until a language-pair model passes evaluation:
# an unverified multilingual model must never produce an unrelated ISL sequence.
MODEL_BY_LANGUAGE = {
    "hi-IN": "Helsinki-NLP/opus-mt-hi-en",
}


class RegionalTranslator:
    def __init__(self, max_models: int = 2):
        self.max_models = max_models
        self._models: OrderedDict[str, tuple[object, object]] = OrderedDict()

    def _load(self, model_name: str):
        cached = self._models.get(model_name)
        if cached is not None:
            self._models.move_to_end(model_name)
            return cached
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).eval()
        self._models[model_name] = (tokenizer, model)
        while len(self._models) > self.max_models:
            self._models.popitem(last=False)
        return tokenizer, model

    def translate(self, text: str, language: str) -> dict[str, str]:
        text = " ".join(str(text).strip().split())
        if not text:
            raise ValueError("text is required")
        if language == "en-IN":
            return {"english": text, "method": "identity"}
        reviewed = REVIEWED_PHRASES.get(language, {}).get(text)
        if reviewed:
            return {"english": reviewed, "method": "reviewed_phrase"}
        model_name = MODEL_BY_LANGUAGE.get(language)
        if not model_name:
            raise ValueError("unsupported language")
        # The product server imports only the reviewed phrase table.  Load
        # PyTorch only when the optional offline neural translator is asked
        # to handle an unreviewed Hindi sentence.
        import torch

        tokenizer, model = self._load(model_name)
        with torch.no_grad():
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=192)
            generated = model.generate(**encoded, max_new_tokens=96, num_beams=4, early_stopping=True)
        english = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        if not english:
            raise RuntimeError("translator produced an empty result")
        return {"english": english, "method": "neural_mt"}
