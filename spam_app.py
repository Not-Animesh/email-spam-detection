import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import tkinter as tk

print("App started")

df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)

model = MultinomialNB()
model.fit(X_train, y_train)

def check_spam():
    msg = entry.get()
    msg_vec = vectorizer.transform([msg])
    prediction = model.predict(msg_vec)

    if prediction[0] == 1:
        result_label.config(text="Spam", fg="red")
    else:
        result_label.config(text="Not Spam", fg="green")

root = tk.Tk()
root.title("Spam Detector")
root.geometry("400x300")

tk.Label(root, text="Enter your message:").pack(pady=10)

entry = tk.Entry(root, width=40)
entry.pack(pady=10)

tk.Button(root, text="Check Spam", command=check_spam).pack(pady=10)

result_label = tk.Label(root, text="")
result_label.pack(pady=20)

root.mainloop()