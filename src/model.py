from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib

def train_models(X, y_reg, y_clf):
    
    reg = RandomForestRegressor(n_estimators=200)
    clf = RandomForestClassifier(n_estimators=200)

    reg.fit(X, y_reg)
    clf.fit(X, y_clf)

    joblib.dump(reg, "models/forecast_model.pkl")
    joblib.dump(clf, "models/classifier_model.pkl")

    return reg, clf