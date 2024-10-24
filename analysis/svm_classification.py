
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from matplotlib import pyplot as plt

# Initialize TF-IDF Vectorizer with more features and n-grams
tfidf = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))

# Initialize LinearSVC outside the function
clf = LinearSVC(C=1.0, max_iter=1000)

# Define file paths outside the function
results_csv_path = 'analysis/svm_result/svm_output'
log_path = "analysis/svm_result/case_narratives_performance.log"
conf_mat_path = "analysis/svm_result/linear_svm_conf_mat.png"

def classify_case_narratives(data_path, test_size=0.50):
    # Read the data into DataFrame
    data = pd.read_csv(data_path)

    # Exclude 'Homeless' and 'Call Sign' columns
    feature_columns = data.columns.difference(['Homeless', 'Call Sign', 'ModeofIntervention'])

    # Split data into X, y
    X, y = data[feature_columns], data['ModeOfIntervention']

    # Merge the text from each column into a single string for each row
    X['Merged'] = X.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

    # Split arrays into random train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X['Merged'], y, test_size=test_size)

    # Use TF-IDF to convert X text data into numerical features
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Train the model on the training dataset
    clf.fit(X_train_tfidf, y_train)

    # Predict on the test dataset
    y_pred = clf.predict(X_test_tfidf)

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
    
    # Convert confusion matrix to DataFrame
    cm_df = pd.DataFrame(cm, index=clf.classes_, columns=clf.classes_)

    # Log performance results and confusion matrix to a .log file
    with open(log_path, 'w') as log_file:
        log_file.write(f"Model accuracy: {accuracy:.4f}\n\n")
        log_file.write("Confusion Matrix:\n")
        log_file.write(cm_df.to_string())  # Write the confusion matrix DataFrame to the log file

    return accuracy

classify_case_narratives('data/2023_CAHOOTS_Call_Data_True_Labels.csv')
