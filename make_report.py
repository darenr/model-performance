import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn import ensemble
from sklearn import model_selection

from evidently import ColumnMapping
from evidently.options import ColorOptions
from evidently.report import Report

from evidently.metrics import ConflictTargetMetric
from evidently.metrics import ConflictPredictionMetric
from evidently.metrics import ClassificationQualityMetric
from evidently.metrics import ClassificationClassBalance
from evidently.metrics import ClassificationConfusionMatrix
from evidently.metrics import ClassificationQualityByClass
from evidently.metrics import ClassificationClassSeparationPlot
from evidently.metrics import ClassificationProbDistribution
from evidently.metrics import ClassificationRocCurve
from evidently.metrics import ClassificationPRCurve
from evidently.metrics import ClassificationPRTable
from evidently.metrics import ClassificationQualityByFeatureTable

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


def load_data():
    bcancer_data = datasets.load_breast_cancer(as_frame="auto")
    bcancer = bcancer_data.frame

    bcancer_ref = bcancer.sample(n=300, replace=False)
    bcancer_cur = bcancer.sample(n=200, replace=False)

    bcancer_label_ref = bcancer_ref.copy(deep=True)
    bcancer_label_cur = bcancer_cur.copy(deep=True)

    model = ensemble.RandomForestClassifier(random_state=1, n_estimators=10)

    model.fit(bcancer_ref[bcancer_data.feature_names.tolist()], bcancer_ref.target)

    bcancer_ref["prediction"] = model.predict_proba(
        bcancer_ref[bcancer_data.feature_names.tolist()]
    )[:, 1]
    bcancer_cur["prediction"] = model.predict_proba(
        bcancer_cur[bcancer_data.feature_names.tolist()]
    )[:, 1]

    bcancer_label_ref["prediction"] = model.predict(
        bcancer_label_ref[bcancer_data.feature_names.tolist()]
    )
    bcancer_label_cur["prediction"] = model.predict(
        bcancer_label_cur[bcancer_data.feature_names.tolist()]
    )

    return bcancer_ref, bcancer_cur, bcancer_label_ref, bcancer_label_cur


def label_binary_classification(model_label_ref, model_label_cur):
    classification_report = Report(
        metrics=[
            ClassificationQualityMetric(),
            ClassificationClassBalance(),
            ConflictTargetMetric(),
            ConflictPredictionMetric(),
            ClassificationConfusionMatrix(),
            ClassificationQualityByClass(),
        ],
        options=[get_color_scheme()],
    )

    classification_report.run(
        reference_data=model_label_ref, current_data=model_label_cur
    )

    return classification_report


def probabilistic_binary_classification(model_ref, model_cur):
    classification_report = Report(
        metrics=[
            ClassificationQualityMetric(),
            ClassificationClassBalance(),
            ConflictTargetMetric(),
            ConflictPredictionMetric(),
            ClassificationConfusionMatrix(),
            ClassificationQualityByClass(),
            ClassificationClassSeparationPlot(),
            ClassificationProbDistribution(),
            ClassificationRocCurve(),
            ClassificationPRCurve(),
            ClassificationPRTable(),
        ],
        options=[get_color_scheme()],
    )

    classification_report.run(reference_data=model_ref, current_data=model_cur)
    return classification_report


def get_evidently_html(evidently_object) -> str:
    """Returns the rendered EvidentlyAI report/metric as HTML"""
    import tempfile
    
    with tempfile.NamedTemporaryFile() as tmp:
        evidently_object.save_html(tmp.name)
        with open(tmp.name, "r", encoding="utf-8") as fh:
            return fh.read()



def get_dp_report(
    evidently_report_label, evidently_report_proba, report_file_name="report.html"
):

    dp_report = dp.Report(
        dp.HTML("<h1>My Classification Report</h1>"),
        dp.Divider(),
        dp.Select(
            dp.HTML(get_evidently_html(evidently_report_label), label="Label Metrics"),
            dp.HTML(
                get_evidently_html(evidently_report_proba),
                label="Probabilistic Metrics",
            ),
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


if __name__ == "__main__":
    model_proba_ref, model_proba_cur, model_label_ref, model_label_cur = load_data()

    evidently_report_proba = probabilistic_binary_classification(
        model_proba_ref, model_proba_cur
    )
    evidently_report_label = label_binary_classification(
        model_label_ref, model_label_cur
    )

    get_dp_report(
        evidently_report_label, evidently_report_proba, report_file_name="report.html"
    )

    print(get_evidently_html(evidently_report_label))