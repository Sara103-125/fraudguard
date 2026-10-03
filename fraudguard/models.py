"""Model definitions. Every model is wrapped in a Pipeline with the scaler.

We use imbalanced-learn's Pipeline instead of scikit-learn's because it can
contain a resampling step (SMOTE). The key property: resampling steps run
only during .fit(), and are skipped during .predict(). So SMOTE only ever
sees training data, never the data we evaluate on.
"""

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from fraudguard.config import RANDOM_STATE
from fraudguard.preprocessing import make_scaler


def build_pipeline(model, use_smote: bool = False) -> Pipeline:
    steps = [("scale", make_scaler())]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("model", model))
    return Pipeline(steps)


def logistic_regression(class_weight=None) -> LogisticRegression:
    # class_weight="balanced" makes each fraud count ~578x more than a normal
    # transaction in the loss, so the model can't just ignore the rare class.
    return LogisticRegression(max_iter=1000, class_weight=class_weight)


def random_forest(class_weight=None) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=2,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def xgboost(scale_pos_weight: float = 1.0) -> XGBClassifier:
    # scale_pos_weight is XGBoost's version of class weights:
    # the usual choice is (number of normal) / (number of fraud).
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def isolation_forest(contamination: float) -> IsolationForest:
    # Unsupervised: it never sees the labels. It learns what "normal" looks like
    # and flags points that are easy to isolate. `contamination` is the share of
    # points it should flag (we set it to the known fraud rate).
    return IsolationForest(
        n_estimators=200,
        contamination=contamination,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
