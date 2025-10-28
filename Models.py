import time
import ollama
from google import genai
from Text import Text
from colorama import Fore, Style, init
import json
import os

# Reset automático de colores tras cada print
init(autoreset=True)


# =====================================
# BASE CLASS
# =====================================
class BaseModel:
    """
    Base class for LLM model wrappers (e.g., Ollama, Gemini).
    Handles shared behavior like metrics tracking, artifact saving,
    and data persistence for prompts and responses.
    """

    def __init__(self, name, supplier, parameters=None):
        safe_name = name.replace(" ", "_").lower()
        self.name = name
        self.supplier = supplier
        self.parameters = parameters or {"temperature": 0.7}
        self.task = "text-generation"
        self.text = Text()
        self.path_artifacts = f"artifacts/{safe_name}_artifacts.json"
        self.path_info = f"info_by_model/{safe_name}_info.json"

        # Metrics
        self.inference_times = []
        self.costs = []
        self.total_tokens = []

    # -------------------------------
    def run_inference(self, prompt):
        raise NotImplementedError("You must implement run_inference in the subclass")

    # -------------------------------
    def record_metrics(self, prompt, answer, inference_time):
        self.inference_times.append(inference_time)
        self.text.set_generate_answer(answer)
        self.costs.append(self.text.cost_estimate(prompt + answer))
        self.total_tokens.append(self.text.count_tokens(prompt + answer))

    # -------------------------------
    def print_result(self, answer, inference_time):
        """
        Prints the model's response line by line with natural flow and color.
        """
        print(Fore.CYAN + "\n" + "─" * 60)
        print(Fore.GREEN + f"🤖 {self.name}: " + Style.RESET_ALL, end="", flush=True)

        # Animación de escritura tipo "streaming"
        for char in answer:
            print(char, end="", flush=True)
            time.sleep(0.005)

        print("\n" + Fore.CYAN + "─" * 60)
        print(Fore.BLUE + f"🕒 Tiempo de inferencia: {inference_time:.2f}s\n")

    # -------------------------------
    def get_parameters(self):
        return {
            "model_name": self.name,
            "supplier": self.supplier,
            "task": self.task,
            **self.parameters,
        }

    # -------------------------------
    def get_metrics(self):
        if not self.inference_times:
            return {}

        return {
            "avg_inference_time": sum(self.inference_times) / len(self.inference_times),
            "total_tokens": sum(self.total_tokens),
            "total_cost_estimate": sum(self.costs),
        }

    # -------------------------------
    def get_artifacts(self):
        if not os.path.exists(self.path_artifacts):
            return []
        with open(self.path_artifacts, "r", encoding="utf-8") as f:
            return json.load(f)

    # -------------------------------
    def set_artifacts(self):
        try:
            if self.text.prompt == "exit":
                return

            last_artifact = {
                "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "prompt": self.text.prompt,
                "answer": self.text.generated_answer,
            }

            os.makedirs("artifacts", exist_ok=True)
            data = self.get_artifacts()
            data.append(last_artifact)
            data = self.text.clean_dict(data)

            with open(self.path_artifacts, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

        except Exception as e:
            print(Fore.RED + f"❌ Error saving artifacts: {e}")

    # -------------------------------
    def reset_artifacts(self):
        try:
            if os.path.exists(self.path_artifacts):
                os.remove(self.path_artifacts)
                print(Fore.GREEN + f"🧹 Artifacts reset for {self.name}")
            else:
                print(Fore.YELLOW + f"⚠️ No artifacts found for {self.name}")
        except Exception as e:
            print(Fore.RED + f"❌ Error resetting artifacts: {e}")

    # -------------------------------
    def show_info_model(self):
        if not os.path.exists(self.path_info):
            print(Fore.RED + "No saved information for this model yet.\n")
            return

        try:
            with open(self.path_info, "r", encoding="utf-8") as f:
                info = json.load(f)

            print(Fore.MAGENTA + "═" * 60)
            print(Fore.MAGENTA + f"📊 INFO FOR MODEL: {self.name.upper()}")
            print(Fore.MAGENTA + "═" * 60 + "\n")

            print(Fore.CYAN + "⚙️ PARAMETERS:")
            for k, v in info.get("parameters", {}).items():
                print(Fore.GREEN + f"  {k}: {v}")

            print(Fore.CYAN + "\n📈 METRICS:")
            for k, v in info.get("metrics", {}).items():
                print(Fore.GREEN + f"  {k}: {v}")

            print(Fore.CYAN + "\n🗂️ ARTIFACTS:")
            for a in info.get("artifacts", []):
                print(Fore.YELLOW + f"  • {a['date']}")
                print(Fore.GREEN + "    🧠 Prompt:" + Style.RESET_ALL, a["prompt"])
                print(Fore.GREEN + "    💬 Answer:" + Style.RESET_ALL, a["answer"])
                print("")

        except Exception as e:
            print(Fore.RED + f"❌ Error loading model info: {e}")

    # -------------------------------
    def save_info(self):
        info = {
            "parameters": self.get_parameters(),
            "metrics": self.get_metrics(),
            "artifacts": self.get_artifacts(),
        }
        os.makedirs("info_by_model", exist_ok=True)
        with open(self.path_info, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=4, ensure_ascii=False)


# =====================================
# OLLAMA MODEL
# =====================================
class OllamaModel(BaseModel):
    def __init__(self, parameters=None):
        super().__init__(name="Ollama", supplier="ollama", parameters=parameters)

    def run_inference(self, prompt):
        try:
            start = time.time()
            response = ollama.chat(
                model="llama3",
                messages=[{"role": "user", "content": prompt}],
                options=self.parameters,
            )
            answer = response["message"]["content"]
            duration = time.time() - start

            self.record_metrics(prompt, answer, duration)
            self.print_result(answer, duration)
            return answer, duration

        except Exception as e:
            print(Fore.RED + f"❌ Error running Ollama inference: {e}")
            return "", 0


# =====================================
# GEMINI MODEL
# =====================================
class GeminiModel(BaseModel):
    def __init__(self, parameters=None):
        super().__init__(name="Gemini", supplier="google", parameters=parameters)
        try:
            self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", "AIzaSyB04ddmfqIpjlE_j2AH3nQVfBV-RE8DMFo"))
        except Exception as e:
            print(Fore.RED + f"⚠️ Gemini client init error: {e}")

    def run_inference(self, prompt):
        try:
            start = time.time()
            response = self.client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt
            )
            answer = response.text
            duration = time.time() - start

            self.record_metrics(prompt, answer, duration)
            self.print_result(answer, duration)
            return answer, duration

        except Exception as e:
            print(Fore.RED + f"❌ Gemini API error: {e}")
            return "", 0


# =====================================
# MODEL FACTORY
# =====================================
class ModelFactory:
    def __init__(self, parameters=None):
        self.parameters = parameters
        self.available_models = {
            "ollama": OllamaModel,
            "gemini": GeminiModel,
        }

    def create(self, model_name):
        model_name = model_name.lower()
        if model_name not in self.available_models:
            print(Fore.RED + f"❌ Model '{model_name}' not available.")
            return None

        model_class = self.available_models[model_name]
        model = model_class(parameters=self.parameters)
        print(Fore.CYAN + f"✅ Model {model.name} initialized successfully.\n")
        return model
