import pandas as pd
from abc import ABC, abstractmethod
from scipy.stats import wasserstein_distance
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import entropy
from scipy.stats import chi2_contingency

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

class WassersteinDistance(Metric):
    def compute(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> float:
        """Compute the Wasserstein Distance between real and synthetic data."""
        return wasserstein_distance(real_data.values.flatten(), synthetic_data.values.flatten())

    @property
    def name(self) -> str:
        return "Wasserstein Distance"

    @property
    def greater_is_better(self) -> bool:
        return False  # Lower distance is better


class MMD(Metric):
    def compute(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> float:
        """Compute the Maximum Mean Discrepancy (MMD) between real and synthetic data using RBF kernel."""
        
        real_kernel = rbf_kernel(real_data, real_data)
        synthetic_kernel = rbf_kernel(synthetic_data, synthetic_data)
        cross_kernel = rbf_kernel(real_data, synthetic_data)

        mmd_value = np.mean(real_kernel) + np.mean(synthetic_kernel) - 2 * np.mean(cross_kernel)
        return mmd_value

    @property
    def name(self) -> str:
        return "Maximum Mean Discrepancy (MMD)"

    @property
    def greater_is_better(self) -> bool:
        return False  # Lower MMD is better
    

class KLDivergenceMetric(Metric):
    def compute(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> float:
        """Compute the Kullback-Leibler (KL) Divergence between real and synthetic data."""

        # Assume the data is in the form of probability distributions (normalized histograms)
        real_data_hist, _ = np.histogram(real_data.values.flatten(), bins=30, density=True)
        synthetic_data_hist, _ = np.histogram(synthetic_data.values.flatten(), bins=30, density=True)

        # Compute KL Divergence (real data vs synthetic data)
        kl_div = entropy(real_data_hist, synthetic_data_hist)
        return kl_div

    @property
    def name(self) -> str:
        return "Kullback-Leibler Divergence (KL)"

    @property
    def greater_is_better(self) -> bool:
        return False  # Lower KL divergence is better
