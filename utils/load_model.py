import joblib

def load_model(path="models/best_norm&model_svm.pkl"):
    return joblib.load(path)
