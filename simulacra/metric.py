import pandas as pd
from abc import ABC, abstractmethod

class Metric(ABC):
    """
    Abstract base class for evaluation metrics.
    """

    @abstractmethod
    def compute(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> float:
        """Abstract method to compute a metric between real and synthetic data.
       
        :param real_data: DataFrame containing the real data.
        :param synthetic_data: DataFrame containing the synthetic data.
        :return: Computed metric as a float.
        """
        pass
 
    @abstractmethod
    def compare(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, selected_model, metric: str ) -> str:
        """Abstract method to compare this metric between model trained on raw data and model trained also with synthetic data.
       
        :param real_data: DataFrame containing the real data.
        :param synthetic_data: DataFrame containing the synthetic data.
        :param selected_model: Model to be evaluated.
        :param metric: Metric to be used for comparison.
        :return: Comparison result as a string.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Abstract method to return the name of the metric.
       
        :return: Name of the metric as a string.
        """
        pass

    @property
    @abstractmethod
    def greater_is_better(self) -> bool:
        """
        Indicates if a higher metric value is better.

        :return: True if higher is better, False otherwise. (bool)
        """
        pass

