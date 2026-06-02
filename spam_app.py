import tkinter as tk
from tkinter import scrolledtext

from model_utils import train_model


model, accuracy = train_model()


def classify_message(message: str) -> tuple[str, float]:
    probabilities = model.predict_proba([message])[0]
    spam_probability = float(probabilities[1])
    label = "Spam" if spam_probability >= 0.5 else "Not Spam"
    return label, spam_probability


def check_spam() -> None:
    message = message_input.get("1.0", tk.END).strip()
    if not message:
        result_label.config(text="Please enter a message.", fg="#E08A00")
        confidence_label.config(text="")
        return

    label, spam_probability = classify_message(message)
    if label == "Spam":
        result_label.config(text="Spam", fg="#D32F2F")
    else:
        result_label.config(text="Not Spam", fg="#2E7D32")
    confidence_label.config(text=f"Spam probability: {spam_probability:.2%}")


def clear_fields() -> None:
    message_input.delete("1.0", tk.END)
    result_label.config(text="")
    confidence_label.config(text="")

root = tk.Tk()
root.title("Email Spam Detector")
root.geometry("520x430")
root.configure(bg="#F5F7FB")

title_label = tk.Label(
    root,
    text="Email Spam Detector",
    font=("Arial", 18, "bold"),
    bg="#F5F7FB",
)
title_label.pack(pady=(15, 5))

subtitle_label = tk.Label(
    root,
    text=f"Model validation accuracy: {accuracy:.2%}",
    font=("Arial", 10),
    fg="#4F5B62",
    bg="#F5F7FB",
)
subtitle_label.pack(pady=(0, 10))

tk.Label(
    root,
    text="Enter your message:",
    font=("Arial", 11),
    bg="#F5F7FB",
).pack(pady=(0, 5))

message_input = scrolledtext.ScrolledText(root, width=60, height=10, wrap=tk.WORD)
message_input.pack(padx=15, pady=(0, 10))

button_frame = tk.Frame(root, bg="#F5F7FB")
button_frame.pack(pady=5)

tk.Button(
    button_frame,
    text="Check Spam",
    command=check_spam,
    bg="#3B82F6",
    fg="white",
    padx=12,
    pady=6,
    relief=tk.FLAT,
).pack(side=tk.LEFT, padx=6)

tk.Button(
    button_frame,
    text="Clear",
    command=clear_fields,
    bg="#E5E7EB",
    padx=12,
    pady=6,
    relief=tk.FLAT,
).pack(side=tk.LEFT, padx=6)

result_label = tk.Label(root, text="", font=("Arial", 14, "bold"), bg="#F5F7FB")
result_label.pack(pady=(15, 5))

confidence_label = tk.Label(root, text="", font=("Arial", 11), bg="#F5F7FB")
confidence_label.pack(pady=(0, 10))

root.mainloop()