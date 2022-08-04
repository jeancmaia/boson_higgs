import os
import shutil
import sys

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from ml_experiments.data import DataHandler 
from ml_experiments.experiment import Experiment
from ml_experiments.model import ModelInterface


def train(
    target_column: str,
    task_type: str,
    dataset_path: Path = "assets/cleaned_data/*",
    output_folder: Path = "assets",
    experiment_name: str = "model",
    metric: str = "roc_auc_ovr_weighted",
    experiment_id: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    random_seed: int = 42,
    n_jobs: int = 3,
    evaluate: bool = True,
    refit: bool = False,
    test_size: float = 0.5
):
    # TODO: to write either method or class for directory handling.
    experiment_output_path = output_folder + "/" + experiment_name
    
    data_manager = DataHandler(
        target_column=target_column, task_type=task_type, dataset_path=dataset_path
    )
    data_manager.load_dataset()
    
    experiment = Experiment(
        data_manager=data_manager,
        model=ModelInterface(),
        evaluate=evaluate,
        random_seed=random_seed,
        experiment_output_path=experiment_output_path,
        n_jobs=n_jobs,
        test_size=test_size
    )
    experiment.run()
    experiment.persist()
    