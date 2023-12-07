import pandas as pd
import numpy as np
from sklearn.metrics import \
    max_error, \
    mean_absolute_error, \
    mean_squared_error, \
    mean_absolute_percentage_error, \
    median_absolute_error, \
    r2_score

import json  # for pretty print dict


class ErrorsToExcel:

    def __init__(self, file_name):
        self.file_name = file_name
        self._container = []

    def add_errors_measurment(self, description: str, true_y, pred_y):
        temp = {
            description: {
                "MaxE":     max_error(true_y, pred_y),
                "MAE":     mean_absolute_error(true_y, pred_y),
                "MSE":     mean_squared_error(true_y, pred_y),
                "RMSE":     mean_squared_error(true_y, pred_y, squared=False),
                "MAPE":     mean_absolute_percentage_error(true_y, pred_y),
                "MedianAE": median_absolute_error(true_y, pred_y),
                "R2_score": r2_score(true_y, pred_y),
            }
        }
        print(json.dumps(temp, indent=4))
        self._container.append(pd.DataFrame(temp).T)

    def write(self):
        df = pd.concat(self._container)
        df.to_excel(f"{self.file_name}.xlsx")


if __name__ == "__main__":
    t1 = np.random.random(10)
    t2 = np.random.random(10)

    errors = ErrorsToExcel(file_name="test")

    errors.add_errors_measurment("test1", t1, t2)
    errors.add_errors_measurment("test2", t1, t2)
    errors.write()

