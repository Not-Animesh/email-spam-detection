import argparse

from model_utils import train_model


def classify_message(model, message: str) -> tuple[str, float]:
    probabilities = model.predict_proba([message])[0]
    spam_probability = float(probabilities[1])
    label = "Spam" if spam_probability >= 0.5 else "Not Spam"
    return label, spam_probability


def main() -> None:
    parser = argparse.ArgumentParser(description="Email spam detection CLI")
    parser.add_argument(
        "--message",
        type=str,
        help="Classify a single message and exit",
    )
    args = parser.parse_args()

    model, accuracy = train_model()
    print(f"Model ready. Validation accuracy: {accuracy:.2%}")

    if args.message is not None:
        message = args.message.strip()
        if not message:
            print("Please provide a non-empty message.")
            return

        label, spam_probability = classify_message(model, message)
        print(f"Prediction: {label} (spam probability: {spam_probability:.2%})")
        return

    while True:
        message = input("Enter your message (or 'quit' to exit): ").strip()

        if message.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        if not message:
            print("Please enter a non-empty message.")
            continue

        label, spam_probability = classify_message(model, message)
        print(f"Prediction: {label} (spam probability: {spam_probability:.2%})")


if __name__ == "__main__":
    main()