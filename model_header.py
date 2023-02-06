import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
import shap

class DfExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.data_frame = None
        self.feature_names_in_ = None
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns)
        return self

    def transform(self, X, y=None):
        self.data_frame = X

        if isinstance(X, pd.DataFrame):
            self.feature_names_out_ = np.array(X.columns)
        return X

    def set_output(self, transform=None):
        return

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


class ModelHousing():
    def __init__(self, model_pipe, mode):
        self.mode = mode
        self.model_pipe = model_pipe
        self.explainer = None
        if mode == 'class':
            self.explainer = shap.TreeExplainer(self.model_pipe[-1].best_estimator_)
        elif mode == 'regr':
            self.explainer = shap.Explainer(self.model_pipe[-1].best_estimator_)

    def make_prediction(self, X, sort_output=True, out_display='perc'):
        out_dict = {}
        y_pred = self.model_pipe.predict(X)
        X_tr = self.model_pipe.named_steps['df_extractor'].data_frame
        shap_vals = self.explainer.shap_values(X_tr)

        if self.mode == 'class':
            pred_shaps = [shap_vals[pred][i] for i, pred in enumerate(y_pred)]
        elif self.mode == 'regr':
            pred_shaps = [shap_vals[i] for i, pred in enumerate(y_pred)]

        for i, (y_pred, shap_vals) in enumerate(zip(y_pred, pred_shaps)):
            if out_display == 'perc':
                shap_vals = np.divide(shap_vals, np.abs(shap_vals).sum()) * 100

            out_shaps = {k: v for k, v in zip(self.model_pipe[:-1].get_feature_names_out(), shap_vals)}

            if sort_output == True:
                out_shaps = sorted(out_shaps.items(), key=lambda x: abs(x[1]), reverse=True)

            pred_dict = {'y_pred': y_pred, 'shap_vals': out_shaps}
            out_dict['row_' + str(i)] = pred_dict

        return out_dict







