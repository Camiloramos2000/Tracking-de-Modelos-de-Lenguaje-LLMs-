import fasttext
import tiktoken
import json
import re


class Text:
    """
    Clase encargada del manejo de texto:
    - Detección de idioma
    - Conteo de tokens
    - Estimación de costo
    - Limpieza y serialización segura de strings/diccionarios
    """

    def __init__(self):
        # Modelo preentrenado de FastText para detección de idioma
        self.model_detect_language = fasttext.load_model("models/lid.176.bin")

        # Tokenizador compatible con modelos OpenAI (cl100k_base)
        self.model_count_tokens = tiktoken.get_encoding("cl100k_base")

        # Estado interno del texto actual
        self.prompt = ""
        self.generated_answer = ""

    # ------------------------------------------------------------
    def detect_language(self, text):
        """
        Detecta el idioma de un texto usando FastText.
        Retorna el código ISO (por ejemplo: 'en', 'es', 'fr').
        """
        lang = self.model_detect_language.predict(text)[0][0]
        lang = lang.replace("__label__", "")
        return lang

    # ------------------------------------------------------------
    def count_tokens(self, text):
        """
        Cuenta el número de tokens en el texto.
        Útil para estimar costos o comparativas entre modelos.
        """
        tokens = self.model_count_tokens.encode(text)
        return len(tokens)

    # ------------------------------------------------------------
    def cost_estimate(self, text):
        """
        Calcula un costo estimado (arbitrario) basado en el número de tokens.
        Multiplica cada token por 0.0001 USD.
        """
        cost = self.count_tokens(text) * 0.0001
        return cost

    # ------------------------------------------------------------
    def set_prompt(self, text):
        """
        Define el prompt actual (entrada del usuario o tarea).
        """
        self.prompt = text.strip()

    # ------------------------------------------------------------
    def set_generate_answer(self, text):
        """
        Define la respuesta generada por el modelo.
        """
        self.generated_answer = text.strip()

    # ------------------------------------------------------------
    def clean_text(self, s):
        """
        Limpia texto de caracteres inválidos o corruptos en UTF-8.
        Esto previene errores al guardar o serializar JSONs.
        """
        if isinstance(s, str):
            return s.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
        return s

    # ------------------------------------------------------------
    def clean_dict(self, d):
        """
        Aplica limpieza recursiva a estructuras anidadas (dicts, listas, strings).
        """
        if isinstance(d, dict):
            return {k: self.clean_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [self.clean_dict(i) for i in d]
        else:
            return self.clean_text(d)
