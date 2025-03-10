import numpy as np
import pandas as pd

from typing import Dict, Any
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

def clean_input_data(data: pd.DataFrame, genome: Dict[str, Any], y: pd.Series) -> np.ndarray:
    """
        Clean input data according to the specifications of the genome, and return as array

        Args:
            data (pd.DataFrame): Input data
            genome (Dict[str, Any]): Genome containing the specifications for the data cleaning
            y (pd.Series): Target variable

        Returns:
            data (np.ndarray): Cleaned data
    """
    data: np.ndarray = np.asarray(data)

    # Apply either feature mask or correlation filter
    if genome["filters"]["use_mask"]:
        data = data[:, np.asarray(genome["filters"]["drop_margins"]).astype('bool')]
    else:
        correlation_matrix: np.ndarray = np.corrcoef(data, y, rowvar=False)
        correlation_to_target: np.ndarray = correlation_matrix[-1, :-1]

        threshold: float = genome["filters"]["correlation_filter"]

        retained_indices: np.ndarray = np.where(np.abs(correlation_to_target) >= threshold)[0]
        data = data[:, retained_indices]

    # Scale the data
    data = scale_data(data, genome["cleaners"]["scaler"], genome["cleaners"]["pca"])
    return data

def scale_data(data: np.ndarray, scaler: str, pca: str) -> np.ndarray:
    """
        Scale input data according to the required scaler and PCA

        Args:
            data (np.ndarray): Input data
            scaler (str): Scaler to use
            pca (str): Whether to use PCA

        Returns:
            data (np.ndarray): Scaled data
    """
    if scaler == "robust":
        scaler = RobustScaler()
        data = scaler.fit_transform(data)
    elif scaler == "standard":
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
    elif scaler == "minmax":
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)

    if pca == "pca":
        pca = PCA(n_components="mle")
        data = pca.fit_transform(data)

    return data