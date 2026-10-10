import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# SWYNEX - Customer Support Ticket Classifier
# Task 3: Model Evaluation
# ==========================================

print("=" * 55)
print("SWYNEX - CUSTOMER SUPPORT TICKET CLASSIFIER")
print("MODEL EVALUATION")
print("=" * 55)

# 1. Load dataset
file_path = Path("data/sample_support_tickets.csv")

if not file_path.exists():
    raise FileNotFoundError(
        f"Dataset not found: {file_path.resolve()}"
    )

data = pd.read_csv(file_path)

# 2. Normalize column names
data.columns = (
    data.columns.astype(str)
    .str.strip()
    .str.lower()
    .str.replace(r"[\s\-]+", "_", regex=True)
)

print("\nAvailable columns:", data.columns.tolist())

# 3. Detect message and category columns
text_options = [
    "text",
    "message",
    "ticket_text",
    "support_message",
    "customer_message",
    "description",
    "ticket",
    "issue"
]

label_options = [
    "category",
    "label",
    "target",
    "ticket_category",
    "issue_type",
    "class"
]

text_col = next(
    (col for col in text_options if col in data.columns),
    None
)

label_col = next(
    (col for col in label_options if col in data.columns),
    None
)

if text_col is None or label_col is None:
    raise ValueError(
        "Could not identify the text and category columns. "
        f"Available columns: {data.columns.tolist()}"
    )

print("Message column:", text_col)
print("Category column:", label_col)

# 4. Clean data
data = data[[text_col, label_col]].dropna().copy()

data[text_col] = data[text_col].astype(str).str.strip()
data[label_col] = data[label_col].astype(str).str.strip()

data = data[
    (data[text_col] != "") &
    (data[label_col] != "")
]

if data[label_col].nunique() < 2:
    raise ValueError("At least two categories are required.")

if len(data) < 4:
    raise ValueError("Add more training examples to the dataset.")

print("\nTotal records:", len(data))
print("\nCategory counts:")
print(data[label_col].value_counts())

X = data[text_col]
y = data[label_col]

# 5. Split dataset
stratify_target = (
    y if y.value_counts().min() >= 2 else None
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=stratify_target
)

# 6. Build model
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        lowercase=True
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

# 7. Train
model.fit(X_train, y_train)

# 8. Evaluate
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n" + "=" * 55)
print("EVALUATION RESULTS")
print("=" * 55)

print(f"\nAccuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

# 9. Test example messages
examples = [
    "I was charged twice for my order",
    "I cannot log into my account",
    "The application keeps crashing",
    "I want to get my money back",
    "My package has not arrived"
]

print("\n" + "=" * 55)
print("SAMPLE PREDICTIONS")
print("=" * 55)

for message in examples:
    prediction = model.predict([message])[0]

    print(f"\nInput: {message}")
    print(f"Predicted Category: {prediction}")

print("\nEvaluation completed successfully!")