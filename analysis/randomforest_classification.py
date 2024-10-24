
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from sklearn.tree import export_graphviz
from graphviz import Source
from matplotlib import pyplot as plt

# Initialize TF-IDF Vectorizer outside the function
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')

# Initialize Random Forest Classifier outside the function
clf = RandomForestClassifier(n_estimators=300, class_weight='balanced')

# Define file paths outside the function
results_csv_path = 'analysis/forest_result/rf_output.csv'
log_path = "analysis/forest_result/case_narratives_performance.log"
conf_mat_path = "analysis/forest_result/random_forest_conf_mat.png"

def classify_case_narratives(data_path, test_size=0.50):
    # Read the data into DataFrame
    data = pd.read_csv(data_path)

    # Merge on axis
    data['Merged'] = data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

    # Split data into X, y
    X, y = data['Merged'], data['ModeOfIntervention']

    # Split arrays into random train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

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

    cm_df = pd.DataFrame(cm, index=clf.classes_, columns=clf.classes_)

    # Log performance results and confusion matrix to a .log file
    with open(log_path, 'w') as log_file:
        log_file.write(f"Model accuracy: {accuracy:.4f}\n\n")
        log_file.write("Confusion Matrix:\n")
        log_file.write(cm_df.to_string())  # Write the confusion matrix DataFrame to the log file

    return accuracy


'''# Export one of the trees from the Random Forest (in this case, tree 0)
tree_number = 0
tree = clf.estimator[tree_number]

# Export the tree as a dot file format
dot_data = export_graphviz(
    tree, 
    out_file=None, 
    feature_names=tfidf.get_feature_names_out(),
    class_names=clf.classes_.astype(str),
    filled=True, 
    rounded=False, 
    special_characters=True
)

# Define path to export tree graph to
tree_path = f"output/random_forest_tree_{tree_number}"

# Use graphviz to display the tree
#graph = Source(dot_data, format='png')
#graph.render(tree_path)
#graph.view()'''
