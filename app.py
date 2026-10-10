
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ============================================
# SWYNEX - Customer Support Ticket Classifier
# Task 3: Intelligent Feature
# ============================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "sample_support_tickets.csv"

CONFIDENCE_THRESHOLD = 0.35


def load_and_train_model():
    """Load the dataset and train the ticket classifier."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Ensure the CSV is inside the data folder."
        )

    data = pd.read_csv(DATA_PATH)

    # Normalize column names
    data.columns = (
        data.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\-]+", "_", regex=True)
    )

    required_columns = {"ticket_text", "category"}

    if not required_columns.issubset(data.columns):
        raise ValueError(
            "Expected CSV columns: ticket_text and category.\n"
            f"Found columns: {data.columns.tolist()}"
        )

    # Clean records
    data = data[["ticket_text", "category"]].dropna().copy()

    data["ticket_text"] = (
        data["ticket_text"].astype(str).str.strip()
    )
    data["category"] = (
        data["category"].astype(str).str.strip()
    )

    data = data[
        (data["ticket_text"] != "")
        & (data["category"] != "")
    ]

    if len(data) < 2 or data["category"].nunique() < 2:
        raise ValueError(
            "The dataset needs at least two categories "
            "and sufficient valid examples."
        )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    model.fit(data["ticket_text"], data["category"])

    return model

def predict_ticket(message, model):
    """Predict a category or recommend human review."""

    message = message.strip()

    if not message:
        return {
            "prediction": "Invalid Input",
            "confidence": None,
            "needs_review": True
        }

    probabilities = model.predict_proba([message])[0]

    best_index = probabilities.argmax()
    confidence = float(probabilities[best_index])
    predicted_category = model.classes_[best_index]

    # Route low-confidence predictions for human review
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "prediction": "Unknown / Needs Human Review",
            "confidence": confidence * 100,
            "needs_review": True
        }

    return {
        "prediction": predicted_category,
        "confidence": confidence * 100,
        "needs_review": False
    }

def display_result(message, result):
    """Display prediction details."""

    print("\nInput:")
    print(message)

    print("\nPrediction:")
    print(result["prediction"])

    if result["confidence"] is not None:
        print(f"\nConfidence: {result['confidence']:.2f}%")

    if result["needs_review"]:
        print("Recommended Action: Human support review")
    else:
        print("Recommended Action: Normal ticket routing")

    print("-" * 55)

def main():
    print("=" * 55)
    print("SWYNEX - CUSTOMER SUPPORT TICKET CLASSIFIER")
    print("Intelligent Prediction and Human Review")
    print("=" * 55)

    try:
        model = load_and_train_model()
        print("\nModel trained successfully.")

    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"\nError: {error}")
        return

    # Example predictions
    examples = [
        "I was charged twice for my order",
        "I cannot log into my account",
        "The application keeps crashing",
        "I want to get my money back",
        "My package has not arrived",
        "I have a completely unusual problem with my account"
    ]

    print("\nSAMPLE PREDICTIONS")
    print("=" * 55)

    for message in examples:
        result = predict_ticket(message, model)
        display_result(message, result)

    # Interactive prediction
    print("\nTRY YOUR OWN CUSTOMER MESSAGE")
    print("Type 'exit' to close the program.")

    while True:
        try:
            message = input("\nEnter customer message: ").strip()

            if message.lower() == "exit":
                print("\nThank you for using SWYNEX!")
                break

            if not message:
                print("Please enter a message.")
                continue

            result = predict_ticket(message, model)
            display_result(message, result)

        except (EOFError, KeyboardInterrupt):
            print("\nProgram closed.")
            break

if __name__ == "__main__":
    main()
    

