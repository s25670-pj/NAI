"""
Problem: Klasyfikacja jakości wina na podstawie danych chemicznych z wykorzystaniem modelu SVM i wizualizacją wyników po redukcji wymiarowości za pomocą PCA.

Autorzy: Szymon Rosztajn, Mateusz Lech

Instrukcja użycia:
1. Upewnij się, że plik 'winequality.csv' znajduje się w tym samym katalogu co ten skrypt.
2. Uruchom skrypt, aby załadować dane, wytrenować model SVM, dokonać predykcji i wyświetlić wyniki.
"""

import warnings
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")


def load_data(file_path: str) -> pd.DataFrame:
    """
    Ładuje dane z pliku CSV.
    """
    return pd.read_csv(file_path, delimiter=';')


def split_data(data: pd.DataFrame) -> tuple:
    """
    Dzieli dane na zestawy treningowe i testowe.
    """
    x = data.drop(columns=["quality"])
    y = data["quality"]
    return train_test_split(x, y, test_size=0.2, random_state=3)


def train_svm(x_train: pd.DataFrame, y_train: pd.Series) -> SVC:
    """
    Trenuje model SVM z jądrem RBF.

    Args:
        x_train (pd.DataFrame): Dane treningowe.
        y_train (pd.Series): Etykiety treningowe.

    Returns:
        SVC: Wytrenowany model SVM.
    """
    model = SVC(kernel='rbf', random_state=3)
    model.fit(x_train, y_train)
    return model


def evaluate_model(model: SVC, x_test: pd.DataFrame, y_test: pd.Series):
    """
    Ocena modelu na podstawie danych testowych.

    Args:
        model (SVC): Wytrenowany model.
        x_test (pd.DataFrame): Dane testowe.
        y_test (pd.Series): Etykiety testowe.

    Returns:
        None: Wyświetla raport klasyfikacyjny i dokładność modelu.
    """
    predictions = model.predict(x_test)
    print(metrics.classification_report(y_test, predictions))
    print("Dokładność modelu:", metrics.accuracy_score(y_test, predictions))


def visualize_with_pca(data: pd.DataFrame, labels: pd.Series):
    """
    Wizualizuje dane po redukcji wymiarowości za pomocą PCA oraz klasyfikuje je modelem SVM.

    Args:
        data (pd.DataFrame): Dane wejściowe.
        labels (pd.Series): Etykiety klasyfikacji.
    """
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(data)

    # Podział danych po PCA
    x_train_pca, x_test_pca, y_train_pca, y_test_pca = train_test_split(x_pca, labels, test_size=0.2, random_state=3)

    # Trening modelu SVM na zredukowanych danych
    svm_pca = SVC(kernel='rbf', random_state=3)
    svm_pca.fit(x_train_pca, y_train_pca)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(x_pca[:, 0], x_pca[:, 1], c=labels, cmap='viridis', s=50, edgecolor='k')
    plt.title("Wizualizacja wyników SVM (PCA)")
    plt.xlabel("Pierwsza główna składowa")
    plt.ylabel("Druga główna składowa")
    plt.colorbar(scatter, label="Jakość wina")
    plt.show()


if __name__ == "__main__":
    data = load_data('winequality.csv')
    x_train, x_test, y_train, y_test = split_data(data)

    svm_model = train_svm(x_train, y_train)
    evaluate_model(svm_model, x_test, y_test)

    visualize_with_pca(data.drop(columns=["quality"]), data["quality"])
