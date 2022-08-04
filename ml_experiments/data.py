import dask.dataframe as dd

from pathlib import Path  

class DataHandler:
    def __init__(self, target_column, task_type, dataset_path):
        self.task_type = task_type
        self.target_column = target_column
        self.dataset_path = dataset_path
        
        self._X = None
        self._y = None
        
    def load_dataset(self) -> None:
        data = dd.read_csv(self.dataset_path)
        self._y = data[self.target_column]
        self._X = data.drop(self.target_column, axis=1)
        if self.task_type == "classification":
            self._y = self._y.astype(int)
            
    @property
    def X(self):
        return self._X
    
    @property
    def y(self):
        return self._y
        