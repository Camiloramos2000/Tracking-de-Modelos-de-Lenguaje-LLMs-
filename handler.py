from Models import ModelFactory
from colorama import Fore, Style, init
import mlflow
import uuid
import os
import pandas as pd
import time
import json
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from mlflow.pyfunc import PythonModel
from mlflow.genai import datasets, evaluate, scorers
from mlflow.genai.scorers import Correctness, Safety
init(autoreset=True)  # Reset automático de colores


class handler_options:

    def __init__(self):
        """Inicializa el manejador con opciones y modelos disponibles."""
        self.options_avaliable = [
            "💬 Chat (con MLflow tracking)",
            "🧠 Evaluate (GenAI MLflow)",
            "📊 Show info (ver información comparativa de los modelos)",
            "🚪 Exit"
        ]
        self.models_avaliable = ModelFactory().available_models

    # ============================================================
    # 🔹 LIMPIAR CONSOLA
    # ============================================================
    def clear_console(self):
        """Limpia la consola según el sistema operativo."""
        os.system('cls' if os.name == 'nt' else 'clear')

    # ============================================================
    # 🔹 MENÚ PRINCIPAL
    # ============================================================
    def show_option_main_menu(self):
        """Muestra el menú principal y devuelve la opción seleccionada."""
        self.clear_console()
        print(Fore.CYAN + "=" * 60)
        print(Style.BRIGHT + Fore.MAGENTA + "🚀 MAIN MENU – LLM Text Generation Interface")
        print(Fore.CYAN + "=" * 60 + "\n")

        for i, option in enumerate(self.options_avaliable):
            print(Fore.YELLOW + f"{i + 1}. {option}")

        print(Fore.CYAN + "\n" + "=" * 60)
        answer = input(Fore.LIGHTYELLOW_EX + "👉 Choose an option (1–4): " + Style.RESET_ALL)

        if not answer:
            print(Fore.RED + "\n⚠️ Option cannot be empty.\n")
            time.sleep(1)
            return None

        try:
            answer = int(answer)
        except ValueError:
            print(Fore.RED + "\n⚠️ Invalid input. Must be a number.\n")
            time.sleep(1)
            return None

        if 1 <= answer <= len(self.options_avaliable):
            self.clear_console()
            return answer
        else:
            print(Fore.RED + "\n⚠️ Option not available.\n")
            time.sleep(1)
            return None

    # ============================================================
    # 🔹 MENÚ DE SELECCIÓN DE MODELO
    # ============================================================
    def show_model_menu(self, title):
        """Permite seleccionar un modelo entre los disponibles."""
        self.clear_console()
        print(Fore.MAGENTA + f"🧩 {title}")
        print(Fore.CYAN + "=" * 50)
        print(Fore.LIGHTYELLOW_EX + "Type 'exit' to return to the main menu\n")

        for i, model in enumerate(self.models_avaliable):
            print(Fore.YELLOW + f" {i + 1}. {model}")

        print("")
        answer = input(Fore.LIGHTYELLOW_EX + "👉 Select model: " + Style.RESET_ALL)

        if answer.lower() == "exit":
            self.clear_console()
            return "exit"

        if not answer:
            print(Fore.RED + "⚠️ Option cannot be empty.\n")
            time.sleep(1)
            return None

        try:
            answer = int(answer)
        except ValueError:
            print(Fore.RED + "⚠️ Invalid input.\n")
            time.sleep(1)
            return None

        if 1 <= answer <= len(self.models_avaliable):
            self.clear_console()
            return list(self.models_avaliable.keys())[answer - 1]
        else:
            print(Fore.RED + "⚠️ Option not available.\n")
            time.sleep(1)
            return None

    # ============================================================
    # 🔹 CHAT CON MLFLOW TRACKING
    # ============================================================
    def chat(self, model):
        """Inicia un chat dinámico, fluido y con registro en MLflow."""
        self.clear_console()
        print(Fore.CYAN + f"🤖 Starting chat with {model.name.upper()}...\n")
        time.sleep(0.5)

        chat_history = []  # Historial de mensajes

        def typing_effect(text, delay=0.015):
            """Simula escritura gradual para respuestas del modelo."""
            for char in text:
                print(char, end="", flush=True)
                time.sleep(delay)
            print()

        def set_stage_and_alias(client, name, version, stage):
            client.transition_model_version_stage(
                name=name, version=version, stage=stage, archive_existing_versions=False
            )
            client.set_registered_model_alias(name=name, alias=stage, version=version)

        class DummyModel(PythonModel):
            def predict(self, context, model_input):
                prompt = model_input.iloc[0]["prompt"]
                return [prompt]

        mlflow.set_experiment("LLMS_Text_Generation")
        with mlflow.start_run(run_name=f"Chat_{model.name}_{uuid.uuid4()}") as run:
            if model.name == "gemini":
                mlflow.gemini.autolog(log_traces = True )

            while True:
                # Mostrar historial
                self.clear_console()
                print(Fore.MAGENTA + f"💬 Chat session with {model.name}\n" + Fore.CYAN + "=" * 60)
                for sender, msg in chat_history[-10:]:
                    color = Fore.YELLOW if sender == "You" else Fore.GREEN
                    print(color + f"{sender}:" + Style.RESET_ALL)
                    print(f"{msg}\n")
                print(Fore.CYAN + "=" * 60)

                prompt = input(Fore.YELLOW + "You: " + Style.RESET_ALL).strip()
                if not prompt:
                    print(Fore.RED + "⚠️ Prompt cannot be empty.")
                    time.sleep(1)
                    continue
                if prompt.lower() in ["exit", "quit"]:
                    print(Fore.CYAN + f"\n👋 Ending chat with {model.name}. Goodbye!\n")
                    break

                model.text.set_prompt(prompt)
                chat_history.append(("You", prompt))

                # Respuesta del modelo con efecto escritura
                print(Fore.GREEN + f"{model.name}:" + Style.RESET_ALL, end=" ")
                answer, inference_time = model.run_inference(prompt)
                typing_effect(answer)
                chat_history.append((model.name, answer))

                print(Fore.BLUE + f"🕓 {inference_time:.2f}s\n")
                model.set_artifacts()

            # Guardar información y registrar modelo en MLflow
            model.save_info()
            print(Fore.GREEN + "\n📦 Logging model data to MLflow...")

            client = MlflowClient()
            name = f"llm_{model.name.lower()}_chat"

            mlflow.log_params(model.get_parameters())
            mlflow.log_metrics(model.get_metrics())
            mlflow.log_artifact(f"artifacts/{model.name.lower()}_artifacts.json")

            input_example = pd.DataFrame({"prompt": [prompt]})
            signature = infer_signature(input_example, [model.text.generated_answer])

            tags = {
                "version": "1.0",
                "model": model.name,
                "supplier": model.supplier,
                "type_model": "LLM",
                "created_by": "Camilo Ramos",
            }

            print(Fore.GREEN + "🤖 Registering model in MLflow...")
            model_info = mlflow.sklearn.log_model(
                sk_model=DummyModel(),
                name=name,
                signature=signature,
                input_example=input_example,
                tags=tags,
            )

            try:
                client.create_registered_model(name=name)
                print(Fore.GREEN + f"✅ Model '{model.name}' successfully registered.")
            except mlflow.exceptions.MlflowException as e:
                if "already exists" in str(e):
                    print(Fore.YELLOW + f"⚠️ Model '{model.name}' already exists.")

            mv = client.create_model_version(
                name=name, source=model_info.model_uri, run_id=run.info.run_id
            )
            client.update_model_version(
                name=mv.name,
                version=mv.version,
                description=f"Version {mv.version} of model {model.name}",
            )
            set_stage_and_alias(client, mv.name, mv.version, "Staging")

            print(Fore.CYAN + f"\n🚀 Model '{mv.name}' now in 'Staging' stage.")
            input(Fore.LIGHTYELLOW_EX + "\n🔙 Press Enter to return...")
            self.clear_console()


                
        # ============================================================
        # 🔹 EVALUAR MODELOS CON MLFLOW GENAI
        # ============================================================
    def evaluate(self, model):
        """Evalúa el rendimiento del modelo con MLflow GenAI evaluators."""
        self.clear_console()
        print(Fore.GREEN + f"🧠 Evaluating model: {model.name}\n")
        time.sleep(0.5)

        mlflow.set_experiment("LLMS_Text_Generation")
        with mlflow.start_run(run_name=f"Eval_{model.name}_{uuid.uuid4()}") as run:
            eval_dataset = pd.DataFrame([
                {
                    "inputs": {
                        "question": "Briefly explain what you can do as an AI agent."
                    },
                    "expectations": {
                        "expected_response": "I assist, reason, generate, and adapt to user goals.",
                        "expected_facts": [
                            "assist users",
                            "reason about problems",
                            "generate text or code",
                            "adapt to context",
                            "help achieve goals"
                        ],
                        "guidelines": "Respond briefly and clearly, summarizing your main agent capabilities."
                    }
                }
            ])
            def predict(question):
                answer, _ = model.run_inference(question)
                return answer

            print(Fore.YELLOW + "⚙️ Running GenAI evaluation...")
            time.sleep(1)
            result = evaluate(
                data=eval_dataset,
                predict_fn=predict,
                scorers=[Correctness()],
            )

            print(Fore.GREEN + "\n✅ Evaluation completed successfully!\n")

        input(Fore.LIGHTYELLOW_EX + "🔙 Press Enter to return...")
        self.clear_console()

    # ============================================================
    # 🔹 MENÚ DE EVALUACIÓN
    # ============================================================
    def evaluate_menu(self):
        """Muestra el submenú de evaluación de modelos."""
        while True:
            self.clear_console()
            print(Fore.MAGENTA + "📊 MODEL EVALUATION MENU")
            print(Fore.CYAN + "=" * 60)
            print(Fore.LIGHTYELLOW_EX + "1️⃣  Evaluate a single model")
            print(Fore.LIGHTYELLOW_EX + "2️⃣  Evaluate all models")
            print(Fore.LIGHTYELLOW_EX + "3️⃣  Back to main menu")
            print(Fore.CYAN + "=" * 60)
            answer = input(Fore.YELLOW + "👉 Choose an option: " + Style.RESET_ALL)
            
            factory = ModelFactory()


            if answer == "1":
                model_name = self.show_model_menu("Selecciona el modelo que quieres evaluar")
                if model_name != "exit" and model_name:             
                    self.evaluate(factory.create(model_name))
                    
            elif answer == "2":
                for model in self.models_avaliable:
                    self.evaluate(model=factory.create(model))
                
            elif answer == "3":
                self.clear_console()
                break
            else:
                print(Fore.RED + "\n⚠️ Invalid option.")
                time.sleep(1)

    # ============================================================
    # 🔹 MOSTRAR INFORMACIÓN DE MODELOS
    # ============================================================
    def show_info_menu(self):
        """Muestra información comparativa de todos los modelos."""
        self.clear_console()
        print(Fore.CYAN + "📄 DISPLAYING MODEL INFORMATION\n" + "=" * 60 + "\n")

        for model_name in self.models_avaliable:
            ModelFactory().create(model_name).show_info_model()

        print("\n" + Fore.CYAN + "=" * 60)
        input(Fore.LIGHTYELLOW_EX + "\n🔙 Press Enter to return...")
        self.clear_console()
