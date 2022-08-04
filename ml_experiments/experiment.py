import logging 

from dask_ml.model_selection import train_test_split
from sklearn.metrics import roc_auc_score 
from ml_experiments.evaluation import ClassificationEvaluation
from pathlib import Path  

RANDOM_STATE=42

class Experiment:
    def __init__(
        self,
        data_manager,
        model,
        evaluate,
        random_seed,
        experiment_output_path,
        n_jobs,
        test_size
    ):        
        self.X = data_manager.X
        self.y = data_manager.y 
        self.model = model 
        self.evaluate = evaluate
        self.random_seed = random_seed 
        self.n_jobs = n_jobs
        self.test_size = test_size
        
        self.X_train = None 
        self.y_train = None 
        self.X_test = None 
        self.y_test = None 
        
        #self.split_evaluate(evaluate)
        self.experiment_output_path = experiment_output_path
        logging.basicConfig(
            filename=self.experiment_output_path + "/experiment.log",
            filemode="w",
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
        )
        
           
    def run(self):
        logging.info("Starting hold-out validation with percentage test size: " + str(self.test_size))
        X_train, X_test, y_train, y_test = self._hold_out()
        logging.info("Holdout split finished succesfully!")
        logging.info("Starting model fitting step")
        self._fit(X_train.compute(), y_train.compute())
        logging.info("XGboost algorithm was fitted successfully")
        if self.evaluate:
            logging.info("Starting validation metrics on train dataset")
            self._evaluation('train_eval',
                            X_train.compute(),
                            y_train.compute()
                            )
            logging.info("Metrics on train dataset crafted successfully")
            logging.info("Starting validation metrics on test dataset")
            self._evaluation('test_eval',
                            X_test.compute(),
                            y_test.compute()
                            )
            logging.info("Metrics on test dataset crafted successfully")
            logging.info("Final evaluation finished succesfully")
        logging.info("Setting the optimal threshold as 0.4")
        self.model.threshold = 0.4
        logging.info("Model training finished succesfully!")
        
    def persist(self):
        self.model.persist(self.experiment_output_path)
        
    def _hold_out(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, 
                                                            self.y, 
                                                            test_size=self.test_size, 
                                                            random_state=self.random_seed)
        logging.info(str("Holdout splitted successfully"))

        return (X_train, X_test, y_train, y_test)
        
    def _evaluation(self, dir, X_test, y_test):
        ce = ClassificationEvaluation(self.experiment_output_path + '/' + dir,
                                      self.model, 
                                      X_test, 
                                      y_test
                                    )
        ce.run()
        logging.info("Classification report has finished succesfully")

        
    def _fit(self, X_train, y_train):
        logging.info("Fitting the model on a dataset w shape: " + str(X_train.shape))
        logging.info("Features Model: " + str(X_train.columns))
        self.model.fit(X_train, y_train)