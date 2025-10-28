from Models import ModelFactory
from colorama import Fore, Style, init
import json
import os
import time
from handler import handler_options

# ========================================
# INITIALIZATION
# ========================================

init(autoreset=True)

def limpiar_consola():
    """Limpia la consola de forma compatible con Windows/Linux."""
    os.system('cls' if os.name == 'nt' else 'clear')


def animacion_bienvenida():
    """Muestra una animación de bienvenida elegante."""
    limpiar_consola()
    mensaje = "🚀 WELCOME TO THE LLM TEXT GENERATION INTERFACE"
    borde = "═" * len(mensaje)
    print(Fore.CYAN + Style.BRIGHT + f"\n{borde}")
    for letra in mensaje:
        print(Fore.CYAN + Style.BRIGHT + letra, end="", flush=True)
        time.sleep(0.02)
    print(f"\n{borde}\n")
    time.sleep(0.6)


def pausa():
    """Pausa elegante antes de volver al menú principal."""
    print(Fore.LIGHTYELLOW_EX + "\n↩️  Presiona ENTER para volver al menú principal...")
    input()


# ========================================
# MAIN EXECUTION BLOCK
# ========================================
if __name__ == "__main__":
    parameters = {"temperature": 0.7}
    factory = ModelFactory(parameters=parameters)
    handler = handler_options()

    animacion_bienvenida()

    while True:
        limpiar_consola()
        option_choosen = handler.show_option_main_menu()

        # Limpieza antes de cada bloque para mantener orden
        if option_choosen == 1:
            limpiar_consola()
            print(Fore.CYAN + "💬 Inicializando modo chat...\n")
            time.sleep(0.6)
            model_name = handler.show_model_menu("Selecciona el modelo con que vas a chatear 🤖")
            if model_name and model_name != "exit":
                limpiar_consola()
                handler.chat(model=factory.create(model_name=model_name))
            pausa()

        elif option_choosen == 2:
            limpiar_consola()
            print(Fore.CYAN + "📊 Iniciando evaluación de modelos con MLflow GenAI...\n")
            time.sleep(0.6)
            handler.evaluate_menu()
            pausa()

        elif option_choosen == 3:
            limpiar_consola()
            print(Fore.CYAN + "📘 Mostrando información comparativa de los modelos...\n")
            time.sleep(0.6)
            handler.show_info_menu()
            pausa()

        elif option_choosen == 4:
            limpiar_consola()
            print(Fore.CYAN + "\n👋 Cerrando sesión... ¡Hasta pronto!\n")
            time.sleep(1)
            break

        else:
            print(Fore.RED + "❌ Opción inválida. Inténtalo de nuevo.")
            pausa()
            limpiar_consola()
            continue
