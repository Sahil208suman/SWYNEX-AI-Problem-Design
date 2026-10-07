import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# Load dataset
data = pd.read_csv("data/sample_support_tickets.csv")

X = data["ticket_text"]
y = data["category"]


# Create ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )),
    ("classifier", LogisticRegression(
        max_iter=1000
    ))
])


# Train the model
model.fit(X, y)


# Example predictions
examples = [
    "I was charged twice for my order",
    "I cannot log into my account",
    "The application keeps crashing",
    "I want to get my money back",
    "My package has not arrived"
]


print("=" * 60)
print("SWYNEX - Customer Support Ticket Classifier")
print("=" * 60)

for message in examples:
    prediction = model.predict([message])[0]

    print("\nInput:")
    print(message)

    print("Predicted Category:")
    print(prediction)

print("\n" + "=" * 60)
print("Prototype completed successfully!")
print("=" * 60)