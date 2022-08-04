<b>HIGGS Data set is a classical among ML pratictioners</b>, which registers 11 milion events tagged either as a signal(boson higgs) or background process. Each event holds 28 related features; 21 kinematic properties measured by the particle detectors, and 7 features calculated on top of the first 21. They might be called as low-level and high-level features, respectivelly.
This is an outstading classification challenge on a real case, hosted on https://archive.ics.uci.edu/ml/datasets/HIGGS and detailed on the paper https://arxiv.org/abs/1402.4735. In a nutshell, the article documents a benchmark among three ML algorithms: Boosted decision trees(BDT), shallow neural networks(NN) and deep neural networks with up to 5 layers(DN); the best results achieved with three models were 0.81, 0.81, 0.88 on ROC AUC respectivelly. The researchers applied a hold-out evaluation with 500k samples for the test data. For more details, do not hesitate to check the paper out.

This project aims to produce a machine learning product for the Boson Higgs detection case, ready to ship to production environments. The project details goes as it follows:

-----
#### 1 - To pull the data

The entry 'data-pulling' in the Makefile triggers downloading execution of the dataset.

```make data-pulling```

-----
#### 2 - EDA

The notebook 'nbks/EDA.ipynb' stores the exploratory data analysis, a statistical analysis and the final proposal for the feature engineering pipeline.

To raise a notebook server:

```make jupyter```

-----
#### 3 - machine learning in action

The notebook 'nbks/Non-linearity_and_model-benchmark' benchmarks some candidate models, runs a hyperparameter search, triggers a comprehensive model evaluation, and enumerates the most importante features alongside relationship with the target.

As a result of this endeavor, the model is set for deploying. The Sklearn Pipeline found must be specified on ml_experiments/model.py::Model; the feature preprocessing is defined on scripts/features_handling.py. We are ready to move on to the next step.

-----
#### 4 - To run a full training

Finally, the complete model is trained by the two commands:

```make data-full-preprocessing```
```make train-model```

The hold-out evaluation genereates logs on both paths: "assets/model/train_eval" e "assets/model/test_eval", where the later provides the definitive evaluation. 
    The roc curve of roc_auc.png reveals the ROC AUC of `0.84`.     
    The calibration curve of calibartion.png reveals a trustworhty prediction.     
    The F1 curve of f1_threshold.png reveals the optimal f1 at `0.4`.   
    The Precision Recall Curve of precision_recall.png finishes the compreehensive report.

At the end, the training log and the model was persisted on "assets/model/experiment.log" e "assets/model/model.joblib"

-----
#### 5 - To analyze the final results interactively.

As long as the model training has succeded, the results might be analyzed on notebook: `nbks/DEFAULT_log_analysis.ipynb`


-----
#### 6 - To deploy model

The API can be raised either as a docker container or local env with poetry. The commands are:

local:     
```make api-server-poetry```    


docker:   
```make docker-build-server```
```make docker-run-server```

-----
#### 7 - To request the model

The notebook 'nbks/API_requests_test.ipynb' has a simple way to request the API.
