from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import f1_score, precision_score, accuracy_score,\
recall_score, roc_curve, auc, roc_auc_score, confusion_matrix
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def split_dataset_X_y (df, df_columns):
    X = df.loc[:, df_columns]
    y = df.loc[:, 'volveria']
    print("X.shape: ", X.shape)
    print("y.shape: ", y.shape, "\n")
    return X, y


def fit_model_grid_search (X, y, model, params):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=117, stratify=y)
    rgscv = GridSearchCV(model, params, scoring='roc_auc', cv=5, return_train_score=True).fit(X_train, y_train)
    print(f"Best score: {rgscv.best_score_}")
    print(f"Best params {rgscv.best_params_}\n")
    return rgscv.best_params_, X_train, X_test, y_train, y_test


def fit_model_random_grid_search (X, y, model, params):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=117)
    rgscv = RandomizedSearchCV(model, params, n_iter=60, scoring='roc_auc', cv=5,
                               return_train_score=True).fit(X_train, y_train)
    print(f"Best score: {rgscv.best_score_}")
    print(f"Best params {rgscv.best_params_}\n")
    return rgscv.best_params_, X_train, X_test, y_train, y_test


def apply_stratified_k_fold (X, y, model):
    kf = StratifiedKFold(n_splits=5)
    test_rocs = []
    for fold_idx, (train_index, test_index) in enumerate(kf.split(X, y)):
        model.fit(X.iloc[train_index], y.iloc[train_index])
        test_roc = roc_auc_score(y.iloc[test_index], model.predict(X.iloc[test_index]))
        print(f"Fold {fold_idx}: AUC ROC score is {test_roc:.4f}")
        test_rocs.append(test_roc)
    print(f"Mean test AUC ROC is: {np.mean(test_rocs):.4f}")
    return model

def get_scores(model, X, y):
    scores = {}
    scores['AUC ROC'] = roc_auc_score(y, model.predict_proba(X)[:,1])
    scores['Accuracy'] = accuracy_score(y, model.predict(X))
    scores['Precision'] = precision_score(y, model.predict(X))
    scores['Recall'] = recall_score(y, model.predict(X), pos_label=0)
    scores['F1-Score'] = f1_score(y, model.predict(X))

    for score in scores:
        print(f"{score} : {scores[score]:.4f}")
    return scores

def plot_confusion_matrix(y_true, y_pred):
    names = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, names)
    df_cm = pd.DataFrame(cm, names, names)

    plt.figure(dpi=100)
    plt.title("Matriz de confusión")
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g', square=True)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

def evaluate_holdout(holdout_full, holdout_df, model):
    y_pred = model.predict(holdout_df)
    y_pred = y_pred.astype(int)
    print("y_pred.shape: ", y_pred.shape)
    print("holdout_df.shape: ", holdout_df.shape, "\n")
    serie_predicha = pd.Series(y_pred, name='volveria')
    serie_predicha = serie_predicha.to_frame()
    df_predicho = holdout_full.join(serie_predicha, how='inner')
    cols = ['id_usuario', 'volveria']
    df_resultado = df_predicho[cols]
    return df_resultado
