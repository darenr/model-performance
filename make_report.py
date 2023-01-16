import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn import ensemble
from sklearn import model_selection

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn_evaluation import ClassifierEvaluator


import datapane as dp


def get_color_scheme():
    return ColorOptions(
        primary_color="#268bd2",
        secondary_color="#073642",
        current_data_color="#268bd2",
        reference_data_color="#073642",
        color_sequence=(
            "#268bd2",
            "#2aa198",
            "#859900",
            "#b58900",
            "#cb4b16",
            "#dc322f",
        ),
    )


def get_report_html(report) -> str:
    """Returns the rendered object report/metric as HTML"""
    import tempfile

    with tempfile.NamedTemporaryFile() as tmp:
        report.save(tmp.name)
        with open(tmp.name) as fh:
            return fh.read()


def generate_datapane_report(report_html, report_file_name="report.html"):

    dp_report = dp.Report(
        dp.HTML("<h1>My Classification Report</h1>"),
        dp.Divider(),
        dp.Select(
            dp.HTML(report_html, label="Label Metrics"),
            dp.HTML("TODO: Add more metrics here", label="TODO"),
            type=dp.SelectType.TABS,
        ),
    ).save(
        path=report_file_name,
        formatting=dp.AppFormatting(
            light_prose=False,
            accent_color="DarkGreen",
            font=dp.FontChoice.SANS,
        ),
    )


def rf_model(X_train, X_test, y_train, y_test) -> ClassifierEvaluator:
    est = RandomForestClassifier(n_estimators=5)
    est.fit(X_train, y_train)
    y_pred = est.predict(X_test)
    y_score = est.predict_proba(X_test)
    feature_list = range(4)
    target_names = ["setosa", "versicolor", "virginica"]

    return ClassifierEvaluator(
        est,
        y_test,
        y_pred,
        y_score,
        feature_list,
        target_names,
        "RandomForestClassifier",
    )


if __name__ == "__main__":
    
    # https://sklearn-evaluation.ploomber.io/en/latest/comparison/report.html

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    generate_datapane_report(
        get_report_html(rf_model(X_train, X_test, y_train, y_test).make_report())
    )
