import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import os
import matplotlib.pyplot as plt
from datetime import datetime 

# Import the classification functions
# from distilbert_classification import classify_case_narratives as classify_distilbert
from svm_classification import classify_case_narratives as classify_tfidf_svm
from randomforest_classification import classify_case_narratives as classify_tfidf_rf



def run_classifier(data_path, classifier_type, repetitions=4):
    total_accuracy = 0
    accuracies = []

    # Run the classifier and accumulate the results

    for _ in range(repetitions):
        if classifier_type == 'svm':
            accuracy = classify_tfidf_svm(data_path)
        elif classifier_type == 'rf':
            accuracy = classify_tfidf_rf(data_path)
        # elif classifier_type == 'distilbert':
        #     accuracy = classify_distilbert(data_path)
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")
        accuracies.append(accuracy)
        total_accuracy += accuracy

    # Calculate the average accuracy

    averaged_accuracy = total_accuracy / repetitions
    # print(accuracies)
    return [averaged_accuracy, accuracies]

def accuracy_hist(accuracies:list, model_name:str, reps:int, avg:float):
    os.makedirs(f'analysis/histograms/{model_name}', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hist_path = f"analysis/histograms/{model_name}/accuracies_hist_{timestamp}.png"
    plt.hist(accuracies)
    plt.title(f"{model_name} accuracy distribution")
    plt.xlabel('accuracy')
    plt.ylabel('frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.2)
    plt.text(0.95, 0.90,f'# of repetitions: {reps}',ha='right', va='top', transform=plt.gca().transAxes, fontsize=10)
    plt.text(0.95, 0.85,f'average: {avg:.3f}',ha='right', va='top', transform=plt.gca().transAxes, fontsize=10)
    plt.savefig(hist_path)
    return

def main():
    # Choose classifier and number of repetitions
    classifier_type = 'rf'
    repetitions = 50

    # Run the classifier, collect avg accuracy and list of accuracies
    averaged_accuracy, accuracy_dist = run_classifier('data/2023_CAHOOTS_Call_Data_True_Labels.csv', classifier_type, repetitions)
    
    #create accuracies histogram
    accuracy_hist(accuracy_dist, classifier_type, repetitions, averaged_accuracy)
    print(f"Averaged Accuracy over {repetitions} runs: {averaged_accuracy:.4f}")

if __name__ == "__main__":
    main()
