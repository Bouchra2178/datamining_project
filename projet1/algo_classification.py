import numpy as np
import pandas as pd
import random
import sklearn
import plotly.express as px
import pandas as pd
import streamlit as st
from statistics import mode as mode2
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import Counter

def accuracy_multiclass(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_samples = len(y_true)
    return correct_predictions / total_samples

def precision_multiclass(y_true, y_pred, average='macro'):
    return sklearn.metrics.precision_score(y_true, y_pred, average=average)

def recall_multiclass(y_true, y_pred, average='macro'):
    return sklearn.metrics.recall_score(y_true, y_pred, average=average)



def precision_multiclass2(y_true, y_pred, average='macro'):
    if average == 'macro':
        class_precisions = [precision_class(y_true, y_pred, label) for label in np.unique(y_true)]
        return np.mean(class_precisions)
    elif average == 'micro':
        true_positives, predicted_positives = 0, 0
        for label in np.unique(y_true):
            true_positives += np.sum((y_true == label) & (y_pred == label))
            predicted_positives += np.sum(y_pred == label)
        return true_positives / predicted_positives if predicted_positives > 0 else 0.0
    elif average == 'weighted':
        class_precisions = [precision_class(y_true, y_pred, label) for label in np.unique(y_true)]
        class_counts = [np.sum(y_true == label) for label in np.unique(y_true)]
        weighted_sum = np.sum([prec * count for prec, count in zip(class_precisions, class_counts)])
        total_samples = np.sum(class_counts)
        return weighted_sum / total_samples if total_samples > 0 else 0.0
    else:
        raise ValueError("Invalid value for 'average'. Use 'macro', 'micro', or 'weighted'.")

def recall_multiclass2(y_true, y_pred, average='macro'):
    if average == 'macro':
        class_recalls = [recall_class(y_true, y_pred, label) for label in np.unique(y_true)]
        return np.mean(class_recalls)
    elif average == 'micro':
        true_positives, actual_positives = 0, 0
        for label in np.unique(y_true):
            true_positives += np.sum((y_true == label) & (y_pred == label))
            actual_positives += np.sum(y_true == label)
        return true_positives / actual_positives if actual_positives > 0 else 0.0
    elif average == 'weighted':
        class_recalls = [recall_class(y_true, y_pred, label) for label in np.unique(y_true)]
        class_counts = [np.sum(y_true == label) for label in np.unique(y_true)]
        weighted_sum = np.sum([recall * count for recall, count in zip(class_recalls, class_counts)])
        total_samples = np.sum(class_counts)
        return weighted_sum / total_samples if total_samples > 0 else 0.0
    else:
        raise ValueError("Invalid value for 'average'. Use 'macro', 'micro', or 'weighted'.")

def f1_score_multiclass2(y_true, y_pred, average='macro'):
    if average == 'macro':
        prec = precision_multiclass(y_true, y_pred, average='macro')
        rec = recall_multiclass(y_true, y_pred, average='macro')
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    elif average == 'micro':
        prec = precision_multiclass(y_true, y_pred, average='micro')
        rec = recall_multiclass(y_true, y_pred, average='micro')
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    elif average == 'weighted':
        prec = precision_multiclass(y_true, y_pred, average='weighted')
        rec = recall_multiclass(y_true, y_pred, average='weighted')
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    else:
        raise ValueError("Invalid value for 'average'. Use 'macro', 'micro', or 'weighted'.")

def precision_class(y_true, y_pred, class_label):
    true_positives = np.sum((y_true == class_label) & (y_pred == class_label))
    predicted_positives = np.sum(y_pred == class_label)
    return true_positives / predicted_positives if predicted_positives > 0 else 0.0

def recall_class(y_true, y_pred, class_label):
    true_positives = np.sum((y_true == class_label) & (y_pred == class_label))
    actual_positives = np.sum(y_true == class_label)
    return true_positives / actual_positives if actual_positives > 0 else 0.0

def specificity_score(y_true, y_pred):
    from sklearn.metrics import multilabel_confusion_matrix
    cm = multilabel_confusion_matrix(y_true, y_pred)
    class_specificities = []
    for i in range(cm.shape[0]):
        TN = cm[i, 0, 0]  # Vrais négatifs
        FP = cm[i, 0, 1]  # Faux positifs
        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0.0
        class_specificities.append(specificity)

    mean_specificity = np.mean(class_specificities)

    return mean_specificity

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        # for leaf node
        self.value = value

class DecisionTreeClassifier():
    def __init__(self, method="h",min_samples_split=2, max_depth=2):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        if method=="h":
            self.method=random.choice(["gini", "entropy","C4.5"])
        else:
            self.method=method
      
    def build_tree(self, dataset, curr_depth=0):
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        
        # split until stopping conditions are met
        if num_samples>self.min_samples_split and curr_depth<=self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # check if information gain is positive
            if best_split["info_gain"]>0:
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["info_gain"])
        
        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset, num_samples, num_features):
        # dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")
        
        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y, self.method)
                    # update the best split if needed
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
                        
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode=="gini":
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        elif mode=="entropy":
            gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        else:
            gain = self.entropy(parent) - (weight_l * self.entropy(l_child) + weight_r * self.entropy(r_child))
            split_info = - (weight_l * np.log2(weight_l) + weight_r * np.log2(weight_r))
            split_info = split_info if split_info != 0 else 1
            gain = gain / split_info
        return gain
    
    def entropy(self, y):
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    def gini_index(self, y):
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
        
    def calculate_leaf_value(self, Y):
        unique_classes, counts = np.unique(Y, return_counts=True)
        return unique_classes[np.argmax(counts)]

    
    def print_tree(self, tree=None, indent=" "):
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)    
    def fit(self, X, Y):
        # if isinstance(Y, pd.DataFrame):
        #     Y = Y.values.flatten()
        # print(Y[:, np.newaxis])
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
        
        
    def predict(self, X):
        predictions = [self.make_prediction(x, self.root) for x in X]
        return predictions

    def make_prediction(self, x, tree):
        if tree.value is not None:
            return tree.value

        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

class RandomForestClassifier:
    def __init__(self, n_trees=10, min_samples_split=2, max_depth=2):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, Y):
        for _ in range(self.n_trees):
            # essaye pour chaque arbre de fair eun pretraitment
            echan=random.randint(400, len(X))
            sample_indices = np.random.choice(len(X), size=echan, replace=False)
            X_sampled = X[sample_indices]
            Y_sampled = Y[sample_indices]
            min=random.randint(1, 3)
            max=random.randint(4,10)
            tree = DecisionTreeClassifier(min_samples_split=min, max_depth=max)
            # tree = DecisionTreeClassifier(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            tree.fit(X_sampled, Y_sampled)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # Utilisez la classe majoritaire comme prévision finale
        # final_predictions = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions)
        final_predictions = np.apply_along_axis(lambda x: np.argmax(np.bincount(x.astype(int))), axis=0, arr=predictions)
        return final_predictions

def data_description2(X,y):
    pca = PCA(2)
    X_projected = pca.fit_transform(X)
    y_flat = y.ravel()
    df = pd.DataFrame({'PC1': X_projected[:, 0], 'PC2': X_projected[:, 1], 'Label': y_flat})
    fig = px.scatter(df, x='PC1', y='PC2', color='Label', title='Scatter Plot with PCA', labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'})
    st.plotly_chart(fig) 

def data_description(X,y):
    pca = PCA(2)
    X_projected = pca.fit_transform(X)
    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]
    fig = plt.figure()
    plt.scatter(x1, x2,c=y, alpha=0.8,cmap='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()
    st.pyplot(fig) 
def performance(model,X_test,Y_test,choix=0):
    df = pd.DataFrame(columns=['EXACTITUDE', 'SPÉCIFICITÉ', 'PRÉCISION', 'RAPPEL', 'F-SCORE'])
    Y_pred = model.predict(X_test) 
    accuracy = accuracy_multiclass(Y_test, Y_pred)
    specifity_score=specificity_score(Y_test,Y_pred)
    precision = precision_multiclass2(Y_test, Y_pred, average='macro')
    recall = recall_multiclass2(Y_test, Y_pred, average='macro')
    f1_score = f1_score_multiclass2(Y_test, Y_pred, average='macro')
    nouvelle_ligne = {'EXACTITUDE': accuracy, 'SPÉCIFICITÉ': specifity_score, 'PRÉCISION': precision, 'RAPPEL': recall, 'F-SCORE': f1_score}
    df = df._append(nouvelle_ligne, ignore_index=True)
    if choix==0:
        cm = confusion_matrix(Y_test, Y_pred)
        class_labels = ["Class 1", "Class 2", "Class 3"]
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels, ax=ax)
        ax.set_xlabel('Prédictions')
        ax.set_ylabel('Vraies valeurs')
        ax.set_title('Matrice de Confusion')
        st.pyplot(fig)
        st.table(df)
    return df

def performance3(X,y,label):
    pca = PCA(2)
    X_projected = pca.fit_transform(X)
    df = pd.DataFrame({'PC1': X_projected[:, 0], 'PC2': X_projected[:, 1], 'Label': 0})
    # Avant le clustering
    fig_before=px.scatter(df, x='PC1', y='PC2', color='Label', title='Before Classification', labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'})
    fig_before.update_traces(marker=dict(size=10, opacity=0.7))
    df = pd.DataFrame({'PC1': X_projected[:, 0], 'PC2': X_projected[:, 1], 'Label': label})
    # Après le clustering
    fig_after=px.scatter(df, x='PC1', y='PC2', color='Label', title='After Classification', labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'})
    fig_after.update_traces(marker=dict(size=10, opacity=0.7))
    # Afficher les deux graphiques
    st.plotly_chart(fig_before)
    st.plotly_chart(fig_after) 
class KNeighborsClassifier:
    def __init__(self, k=5, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values
        predictions = []
        for x in X_test:
            distances = self.get_distance(x)
            indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[indices]
            most_common_label = self._most_common(k_nearest_labels)
            predictions.append(most_common_label)
        return predictions
    def get_distance(self, x):
        distance_functions = {
            'manhattan': self._distance_manhattan,
            'euclidean': self._distance_euclidean,
            'minkowski': self._distance_minkowski,
            'cosine': self._distance_cosine,
            'hamming': self._distance_hamming
        }
        return distance_functions.get(self.distance_metric, lambda a, b: None)(x, self.X_train)

    def _distance_manhattan(self, A, B):
        return np.sum(np.abs(A - B), axis=1)

    def _distance_euclidean(self, A, B):
        return np.sqrt(np.sum((A - B)**2, axis=1))

    def _distance_minkowski(self, A, B):
        # Assuming p is 2 for Euclidean distance
        return np.power(np.sum(np.power(np.abs(A - B), 2), axis=1), 1/2)

    def _distance_cosine(self, A, B):
        dot_product = np.dot(A, B.T)
        norm_A = np.linalg.norm(A, axis=1)
        norm_B = np.linalg.norm(B, axis=1)
        return 1 - dot_product / (norm_A * norm_B)

    def _distance_hamming(self, A, B):
        return np.sum(np.array(list(A)) != np.array(list(B)), axis=1)

    def _most_common(self, lst):
        data = Counter(lst)
        return data.most_common(1)[0][0]

