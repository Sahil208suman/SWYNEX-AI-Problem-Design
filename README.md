# SWYNEX – AI Problem Design

## Task 1: AI Problem Design

### Project Title
AI-Based Customer Support Ticket Classification

## 1. Problem Statement

Customer-support teams receive a large number of customer queries every day.
Manually categorizing these queries can be time-consuming and inconsistent.

The proposed AI system will automatically classify customer-support messages
into predefined categories.

The categories are:

- Payment Issue
- Account Issue
- Technical Issue
- Refund Request
- Delivery Issue

The goal is to reduce manual categorization effort and provide consistent
classification of incoming support requests.

## 2. Target User

The target users are:

- Customer-support teams
- Customer-service representatives
- Support operations teams
- Small and medium-sized businesses

## 3. AI Use Case

This is a supervised text-classification problem.

### Input

A customer-support message.

Example:

"I was charged twice for the same transaction."

### Expected Output

Payment Issue

## 4. Data Source

The prototype will use a small labeled dataset of customer-support messages.

Each record will contain:

| Field | Description |
|---|---|
| ticket_text | Customer support message |
| category | Correct support category |

The initial dataset can contain representative synthetic examples for
prototyping.

## 5. Proposed AI Approach

The proposed machine-learning pipeline is:

Customer Message
↓
Text Cleaning
↓
TF-IDF Vectorization
↓
Classification Model
↓
Predicted Category

The initial baseline model will use TF-IDF with Logistic Regression.

## 6. Constraints

1. The initial dataset will be relatively small.
2. Customer messages may contain spelling mistakes.
3. Some messages may be ambiguous.
4. Categories may have unequal numbers of examples.
5. Sensitive customer information should not be exposed.
6. Human review should remain available for uncertain predictions.

## 7. Evaluation Approach

The dataset will be divided into:

- 80% training data
- 20% testing data

The model will be evaluated on previously unseen test data.

The evaluation metrics will include:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## 8. Success Criteria

The initial prototype targets are:

- Accuracy >= 80%
- F1 Score >= 0.75

These are prototype targets and should be validated using a representative
dataset.

## 9. Risks and Limitations

The system may perform poorly when:

- A message contains very little information.
- A message belongs to multiple categories.
- New types of issues appear.
- The training dataset is not representative.
- Categories are highly imbalanced.

## 10. Expected Outcome

The expected outcome is a documented AI problem definition and evaluation
framework for automatically classifying customer-support tickets.

The design can later be extended into a complete machine-learning application.

## 11. Future Improvements

Future versions could include:

- A larger dataset
- More support categories
- Transformer-based NLP models
- Confidence scores
- Human feedback
- Model monitoring
- Integration with customer-support software

## 12. Conclusion

This project defines a practical and measurable AI problem using text
classification.

It clearly defines the target users, data requirements, AI approach,
constraints, evaluation metrics, and success criteria.


## Task 2: Model / API Integration

### Prototype

The AI problem defined in Task 1 was implemented as a working
machine-learning prototype.

### Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression

### Architecture

Customer Support Message
↓
TF-IDF Vectorization
↓
Logistic Regression
↓
Predicted Support Category

### Example Results

| Input | Predicted Category |
|---|---|
| I was charged twice for my order | Payment Issue |
| I cannot log into my account | Account Issue |
| The application keeps crashing | Technical Issue |
| I want to get my money back | Refund Request |
| My package has not arrived | Delivery Issue |

### How to Run

Install dependencies:

```bash
pip install -r requirements.txt
# SWYNEX – AI Problem Design

## Customer Support Ticket Classifier

### Project Overview

SWYNEX is a machine learning prototype that classifies customer support messages into categories such as Account Issue, Payment Issue, Technical Issue, Refund Request, and Delivery Issue.

The system also includes confidence-based prediction handling and human review for uncertain or unfamiliar messages.

### Features

* Customer support ticket classification
* Text preprocessing using TF-IDF
* Logistic Regression classification model
* Prediction confidence display
* Unknown or low-confidence message handling
* Human support review recommendations
* Model evaluation and sample predictions

### Technology Stack

* Python
* Pandas
* Scikit-learn
* TF-IDF Vectorization
* Logistic Regression

### Project Structure

```text
SWYNEX-AI-Problem-Design/
├── Data/
│   └── sample_support_tickets.csv
├── examples/
│   └── example_predictions.txt
├── app.py
├── evaluation.py
├── requirements.txt
└── README.md
```

### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Run the Classifier

```bash
python app.py
```

### Run Model Evaluation

```bash
python evaluation.py
```

### Example Input

`I was charged twice for my order`

Expected category: `Payment Issue`

### Human Review

Messages that are unfamiliar or have insufficient prediction confidence are routed for human review. The confidence threshold should be validated using suitable evaluation data before deployment.

### Limitations

* Performance depends on the size and quality of the training dataset.
* Predictions may be unreliable for unfamiliar messages.
* The model is a prototype and should not replace human support decisions.

### Task 3: Intelligent Feature

This project demonstrates a basic intelligent classification feature with error handling, evaluation examples, and human review recommendations.

### Future Improvements

* Expand the labeled training dataset.
* Improve model evaluation with a larger test set.
* Build a Streamlit web interface.
* Add feedback collection to improve future predictions.

### Author

Sahil Suman

### License

This project is for educational and prototype purposes.

