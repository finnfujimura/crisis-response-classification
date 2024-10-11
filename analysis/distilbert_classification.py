# DistilBERT Random Forest Classification
# Script by Aussie Frost
# Updated on Sept 18, 2024

import numpy as np
import pandas as pd

from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
import torch
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from sklearn.tree import export_graphviz
from graphviz import Source
from matplotlib import pyplot as plt

# Load DistilBERT tokenizer and model outside the function
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Define text encoder outside the function
def text_encoder(texts, tokenizer, model):
    encoded_inputs = tokenizer(texts.tolist(), padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoded_inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Initialize Random Forest Classifier outside the function
clf = RandomForestClassifier(n_estimators=100, class_weight='balanced')

# Define file paths outside the function
results_csv_path = 'analysis/distilbert_result/distilbert_output.csv'
log_path = "analysis/distilbert_result/case_narratives_performance.log"
conf_mat_path = "analysis/distilbert_result/random_forest_conf_mat.png"

def classify_case_narratives(data_path, test_size=0.50):
    # Read the data into DataFrame
    data = pd.read_csv(data_path)

    # Merge on axis
    data['Merged'] = data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

    # Split data into X, y
    X, y = data['Merged'], data['ModeOfIntervention']

    # Split arrays into random train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Tokenize the text data and convert to tensors
    X_train_distilbert = text_encoder(X_train, tokenizer, model)
    X_test_distilbert = text_encoder(X_test, tokenizer, model)

    # Train the model on the training dataset
    clf.fit(X_train_distilbert, y_train)

    # Predict on the test dataset
    y_pred = clf.predict(X_test_distilbert)

    # Determine accuracy metrics
    accuracy = accuracy_score(y_test, y_pred)

    # Create new DF with test results
    results = pd.DataFrame({
        'X': X_test,
        'y_true': y_test,
        'y_pred': y_pred
    })

    # Output the test results to a CSV
    results.to_csv(results_csv_path)

    # Create and plot a confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=clf.classes_.astype(str))
    disp.plot(cmap='viridis')
    plt.savefig(conf_mat_path)

    cm_df = pd.DataFrame(cm, index=clf.classes_, columns=clf.classes_)

    # Log performance results and confusion matrix to a .log file
    with open(log_path, 'w') as log_file:
        log_file.write(f"Model accuracy: {accuracy:.4f}\n\n")
        log_file.write("Confusion Matrix:\n")
        log_file.write(cm_df.to_string())  # Write the confusion matrix DataFrame to the log file
    
    return accuracy