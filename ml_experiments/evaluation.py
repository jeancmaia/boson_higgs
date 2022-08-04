import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (roc_curve, auc, RocCurveDisplay, PrecisionRecallDisplay, f1_score)
from sklearn.calibration import CalibrationDisplay
from abc import ABC   

class Evaluation(ABC):
    pass  


class ClassificationEvaluation(Evaluation):
    def __init__(self, location, model, X_test, y_test):
        self.location = location
        self.model = model 
        self.X_test = X_test
        self.y_test = y_test
        self.y_proba = model.predict_proba(self.X_test)[:,1]
        
    def _rocauc(self):
        plt.clf()
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_proba)
        roc_auc = auc(fpr, tpr)
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='XGBoost test ROC')
        display.plot()
        plt.savefig(self.location+"/roc_auc.png")
        plt.close()
        
    def _precisionrecall(self):
        plt.clf()
        disp = PrecisionRecallDisplay.from_predictions(self.y_test, self.y_proba)
        disp.plot()
        plt.savefig(self.location+"/precision_recall.png")
        plt.close()
               
    def _f1_thresholds(self):
        plt.clf()
        f1_scores = []
        thresholds = np.arange(0, 1, 0.025)
        for threshold in thresholds:
            y_pred = (self.y_proba >= threshold).astype(int)
            f1_scores.append(f1_score(self.y_test, y_pred))
    
        neat_f1 = f1_scores.index(max(f1_scores))

        plt.title('F1-score curve')
        plt.plot(thresholds, f1_scores,  marker='o')
        plt.vlines(thresholds[neat_f1], ymin=0, ymax=max(f1_scores), 
                linestyles='dashed',
                colors='red')
        plt.xlabel('threshold')
        plt.ylabel('f1_micro') 
        plt.savefig(self.location+"/f1_threshold.png")
        plt.close()
        
    def _calibration(self):
        plt.clf()
        disp = CalibrationDisplay.from_predictions(self.y_test, self.y_proba, n_bins=15, name='XGBoost test')
        plt.savefig(self.location+"/calibration.png")
        plt.close()
        
    def run(self):
        self._rocauc()
        self._precisionrecall()
        self._f1_thresholds()
        self._calibration()
        
        