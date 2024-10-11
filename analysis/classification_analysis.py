import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import os

# Import the classification functions
from distilbert_classification import classify_case_narratives as classify_distilbert
from svm_classification import classify_case_narratives as classify_tfidf_svm
from randomforest_classification import classify_case_narratives as classify_tfidf_rf

def run_classifier(data_path, classifier_type, repetitions=1):
    total_accuracy = 0

    # Run the classifier and accumulate the results
    for _ in range(repetitions):
        if classifier_type == 'svm':
            accuracy = classify_tfidf_svm(data_path)
        elif classifier_type == 'rf':
            accuracy = classify_tfidf_rf(data_path)
        elif classifier_type == 'distilbert':
            accuracy = classify_distilbert(data_path)
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")

        total_accuracy += accuracy

    # Calculate the average accuracy
    averaged_accuracy = total_accuracy / repetitions
    return averaged_accuracy

def main():
    # Choose classifier and number of repetitions
    classifier_type = 'svm'
    repetitions = 5

    # Run the classifier
    averaged_accuracy = run_classifier('data/2023_CAHOOTS_Call_Data_True_Labels.csv', classifier_type, repetitions)
    print(f"Averaged Accuracy over {repetitions} runs: {averaged_accuracy:.4f}")

if __name__ == "__main__":
    main()
