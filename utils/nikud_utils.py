from phonikud import Phonemizer

def add_nikud_to_text(text: str) -> str:
    """Add pronunciation nikud to Hebrew text"""
    try:
        phonemizer = Phonemizer()
        rezultat = phonemizer.phonemize(
            text,
            True,  # preserve_punctuation
            True,  # preserve_stress
            False,  # use_expander
            True,  # use_post_normalize
            True,  # predict_stress
            True,  # predict_vocal_shva
            'post',  # stress_placement
            'nikud'  # schema
        )
        return rezultat
    except Exception as e:
        # If phonikud fails, return original text
        print(f"Nikud processing failed: {e}")
        return text

# Phonikud phonemizes the text with IPA or Hebrew nikud.

# Assuming it returns text with vowels.
