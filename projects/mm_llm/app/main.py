import models  

from core.registry import list_models
from app.engine import ChatEngine

def main():
    models = list_models()

    print("\nAvailable Models:")
    for i, m in enumerate(models):
        print(f"{i}. {m}")

    choice = int(input("\nSelect model: "))
    model_name = models[choice]

    engine = ChatEngine(model_name)
    engine.chat()

if __name__ == "__main__":
    main()
