# ========================= Import Necessary Libraries =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr, kendalltau
from lifelines import KaplanMeierFitter, CoxPHFitter
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta
import warnings
from tqdm import tqdm
import random

# ========================= Setup and Configuration =========================

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
def set_seed(seed=42):
    """
    Sets the random seed for reproducibility.
    
    Parameters:
    - seed (int): The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# ========================= Clustering Analysis =========================

class ClusteringAnalysis:
    def __init__(self, data: pd.DataFrame, feature_columns: list, customer_id_column: str = 'customer_id'):
        """
        Initializes the ClusteringAnalysis class with features for clustering.
        
        Parameters:
        - data (pd.DataFrame): DataFrame containing customer data, including 'customer_id' and other features.
        - feature_columns (list of str): List of column names to use for clustering.
        - customer_id_column (str): Column name for customer IDs.
        """
        self.df = data.copy()
        self.feature_columns = feature_columns
        self.customer_id_column = customer_id_column
        self.scaled_features = None  # To store scaled features used for clustering

        # Verify that the selected features exist in the DataFrame
        missing_features = [feature for feature in self.feature_columns if feature not in self.df.columns]
        if missing_features:
            raise ValueError(f"The following feature columns are missing from the data: {missing_features}")

    def _scale_features(self):
        """
        Scales the selected features using StandardScaler.
        This is crucial for algorithms like KMeans and DBSCAN that are sensitive to feature scaling.
        """
        scaler = StandardScaler()
        self.scaled_features = scaler.fit_transform(self.df[self.feature_columns])

    def kmeans_cluster(self, n_clusters: int = 3, random_state: int = 42, visualize: bool = True):
        """
        Applies KMeans clustering to the data.
        
        Parameters:
        - n_clusters (int): The number of clusters to form.
        - random_state (int): Determines random number generation for centroid initialization.
        - visualize (bool): Whether to visualize the clustering results.
        
        Returns:
        - pd.DataFrame: DataFrame with an additional 'cluster_kmeans' column indicating cluster assignments.
        """
        self._scale_features()
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.df['cluster_kmeans'] = kmeans.fit_predict(self.scaled_features)
        
        if visualize:
            self.plot_clusters('cluster_kmeans', title='KMeans Clustering Results')
        
        return self.df.copy()

    def dbscan_cluster(self, eps: float = 0.5, min_samples: int = 5, visualize: bool = True):
        """
        Applies DBSCAN clustering to the data.
        
        Parameters:
        - eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        - min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        - visualize (bool): Whether to visualize the clustering results.
        
        Returns:
        - pd.DataFrame: DataFrame with an additional 'cluster_dbscan' column indicating cluster assignments.
        """
        self._scale_features()
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.df['cluster_dbscan'] = dbscan.fit_predict(self.scaled_features)
        
        if visualize:
            self.plot_clusters('cluster_dbscan', title='DBSCAN Clustering Results')
        
        return self.df.copy()

    def hierarchical_cluster(self, n_clusters: int = 3, linkage: str = 'ward', visualize: bool = True):
        """
        Applies Hierarchical clustering to the data.
        
        Parameters:
        - n_clusters (int): The number of clusters to find.
        - linkage (str): Linkage criterion. Supported options are 'ward', 'complete', 'average', 'single'.
        - visualize (bool): Whether to visualize the clustering results.
        
        Returns:
        - pd.DataFrame: DataFrame with an additional 'cluster_hierarchical' column indicating cluster assignments.
        """
        self._scale_features()
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        self.df['cluster_hierarchical'] = hierarchical.fit_predict(self.scaled_features)
        
        if visualize:
            self.plot_clusters('cluster_hierarchical', title='Hierarchical Clustering Results')
        
        return self.df.copy()

    def plot_clusters(self, cluster_column: str, title: str = 'Clustering Results', highlight_ids: list = None):
        """
        Plots a scatter plot of the clustering results.
        
        Parameters:
        - cluster_column (str): The column name containing cluster labels.
        - title (str): The title of the plot.
        - highlight_ids (list or None): List of customer IDs to highlight in the plot.
        """
        plt.figure(figsize=(10, 7))
        unique_clusters = self.df[cluster_column].unique()
        palette = sns.color_palette('viridis', n_colors=len(unique_clusters))
        
        sns.scatterplot(
            data=self.df,
            x=self.feature_columns[0],
            y=self.feature_columns[1],
            hue=cluster_column,
            palette=palette,
            s=50,
            alpha=0.6,
            edgecolor='k'
        )
        
        # Highlight specific customers if provided
        if highlight_ids is not None:
            highlight_data = self.df[self.df[self.customer_id_column].isin(highlight_ids)]
            sns.scatterplot(
                data=highlight_data,
                x=self.feature_columns[0],
                y=self.feature_columns[1],
                color='red',
                s=200,
                marker='X',
                label='Highlighted Customers',
                edgecolor='k'
            )
        
        plt.title(title)
        plt.xlabel(self.feature_columns[0].replace('_', ' ').title())
        plt.ylabel(self.feature_columns[1].replace('_', ' ').title())
        plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# ========================= Correlation Analysis =========================

class CorrelationAnalyzer:
    """
    A class for analyzing correlations between columns in a DataFrame.
    Supports Pearson, Spearman, and Kendall correlation coefficients.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the CorrelationAnalyzer with a DataFrame.
        
        Parameters:
        - df (pd.DataFrame): DataFrame containing the data to analyze.
        """
        self.df = df.copy()
        self.pearson_corr = None
        self.spearman_corr = None
        self.kendall_corr = None

    def calculate_pearson(self):
        """
        Calculates the Pearson correlation matrix.
        """
        self.pearson_corr = self.df.corr(method='pearson')
        print("Pearson correlation matrix calculated.")

    def calculate_spearman(self):
        """
        Calculates the Spearman correlation matrix.
        """
        self.spearman_corr = self.df.corr(method='spearman')
        print("Spearman correlation matrix calculated.")

    def calculate_kendall(self):
        """
        Calculates the Kendall correlation matrix.
        """
        self.kendall_corr = self.df.corr(method='kendall')
        print("Kendall correlation matrix calculated.")

    def visualize_correlation(self, method: str = 'pearson', figsize: tuple = (10, 8), annot: bool = True, cmap: str = 'coolwarm'):
        """
        Visualizes the correlation matrix using a heatmap.
        
        Parameters:
        - method (str): The correlation method to visualize ('pearson', 'spearman', 'kendall').
        - figsize (tuple): The size of the figure.
        - annot (bool): Whether to annotate the heatmap with correlation coefficients.
        - cmap (str): The color map to use for the heatmap.
        """
        if method == 'pearson':
            if self.pearson_corr is None:
                self.calculate_pearson()
            corr = self.pearson_corr
            title = 'Pearson Correlation Matrix'
        elif method == 'spearman':
            if self.spearman_corr is None:
                self.calculate_spearman()
            corr = self.spearman_corr
            title = 'Spearman Correlation Matrix'
        elif method == 'kendall':
            if self.kendall_corr is None:
                self.calculate_kendall()
            corr = self.kendall_corr
            title = 'Kendall Correlation Matrix'
        else:
            raise ValueError("Invalid method. Choose 'pearson', 'spearman', or 'kendall'.")

        plt.figure(figsize=figsize)
        sns.heatmap(corr, annot=annot, fmt=".2f", cmap=cmap, linewidths=0.5)
        plt.title(title)
        plt.show()

    def pairwise_correlation(self, method: str = 'pearson') -> pd.DataFrame:
        """
        Computes pairwise correlation coefficients and p-values between variables.

        Parameters:
        - method (str): The method to use for correlation ('pearson', 'spearman', 'kendall').

        Returns:
        - pd.DataFrame: A DataFrame containing pairs of variables, their correlation coefficients, and p-values.
        """
        cols = self.df.columns
        results = []
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                var1 = cols[i]
                var2 = cols[j]
                if method == 'pearson':
                    corr, p = pearsonr(self.df[var1], self.df[var2])
                elif method == 'spearman':
                    corr, p = spearmanr(self.df[var1], self.df[var2])
                elif method == 'kendall':
                    corr, p = kendalltau(self.df[var1], self.df[var2])
                else:
                    raise ValueError("Invalid method. Choose 'pearson', 'spearman', or 'kendall'.")
                results.append({'Variable 1': var1, 'Variable 2': var2, 'Correlation': corr, 'P-value': p})
        return pd.DataFrame(results)

    def plot_pairwise_relationships(self, kind: str = 'scatter', hue: str = None, figsize: tuple = (12, 10)):
        """
        Plots pairwise relationships between variables using Seaborn's pairplot.
        
        Parameters:
        - kind (str): The kind of plot to use ('scatter', 'reg', 'kde', 'hist').
        - hue (str): The variable name to use for color encoding.
        - figsize (tuple): The size of the figure.
        """
        sns.pairplot(self.df, kind=kind, hue=hue, diag_kind='kde', corner=True)
        plt.show()

# ========================= Regression Analysis =========================

class TimeSeriesDataset(Dataset):
    """
    Custom Dataset for Time Series data to be used with PyTorch DataLoader.
    """
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class TransformerRegressor(nn.Module):
    """
    Transformer-based model for regression tasks.
    """
    def __init__(self, input_size: int = 1, d_model: int = 128, nhead: int = 8, num_encoder_layers: int = 4, dim_feedforward: int = 256, dropout: float = 0.1, max_seq_length: int = 5000):
        super(TransformerRegressor, self).__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.input_linear = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_length)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.decoder = nn.Linear(d_model, 1)
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights for the linear layers.
        """
        initrange = 0.1
        self.input_linear.weight.data.uniform_(-initrange, initrange)
        self.input_linear.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()

    def forward(self, src):
        """
        Forward pass for the Transformer model.
        """
        # src shape: (batch_size, sequence_length, input_size)
        src = self.input_linear(src) * np.sqrt(self.d_model)  # (batch_size, sequence_length, d_model)
        src = self.pos_encoder(src)  # (batch_size, sequence_length, d_model)
        
        # Transformer expects input of shape (sequence_length, batch_size, d_model)
        src = src.permute(1, 0, 2)  # (sequence_length, batch_size, d_model)
        output = self.transformer_encoder(src)  # (sequence_length, batch_size, d_model)
        output = output.permute(1, 0, 2)  # (batch_size, sequence_length, d_model)
        
        output = self.decoder(output)  # (batch_size, sequence_length, 1)
        
        # Select the last time step's output
        output = output[:, -1, :]  # (batch_size, 1)
        return output

class PositionalEncoding(nn.Module):
    """
    Positional Encoding module injects information about the relative or absolute position
    of the tokens in the sequence.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

class RegressionAnalysis:
    def __init__(self, data: pd.DataFrame, target: str):
        """
        Initializes the regression analysis class.
    
        Parameters:
        - data (pd.DataFrame): DataFrame containing all features and the target variable.
        - target (str): The target variable column name.
        """
        self.data = data.copy()
        self.target = target
        self.X = None
        self.y = None
        self.models = {}
        self.results = {}

    def set_features(self, feature_columns: list):
        """
        Sets the predictor and target variables.
    
        Parameters:
        - feature_columns (list of str): List of predictor variable column names.
        """
        self.X = self.data[feature_columns]
        self.y = self.data[self.target]
        print(f"Selected features: {feature_columns}")
        print(f"Target: {self.target}")

    def train_test_split_data(self, test_size: float = 0.2, random_state: int = 42):
        """
        Splits the data into training and testing sets.
    
        Parameters:
        - test_size (float): Proportion of the dataset to include in the test split.
        - random_state (int): Seed used by the random number generator.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        print(f"Training samples: {self.X_train.shape[0]}")
        print(f"Testing samples: {self.X_test.shape[0]}")

    def train_ridge(self, alpha: float = 1.0):
        model = Ridge(alpha=alpha)
        model.fit(self.X_train, self.y_train)
        self.models['Ridge'] = model
        print("Ridge regression trained.")

    def train_lasso(self, alpha: float = 0.1):
        model = Lasso(alpha=alpha)
        model.fit(self.X_train, self.y_train)
        self.models['Lasso'] = model
        print("Lasso regression trained.")

    def train_svm(self, C: float = 1.0, kernel: str = 'rbf'):
        model = SVR(C=C, kernel=kernel)
        model.fit(self.X_train, self.y_train)
        self.models['SVM'] = model
        print("SVM regression trained.")

    def train_random_forest(self, n_estimators: int = 100, random_state: int = 42):
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        model.fit(self.X_train, self.y_train)
        self.models['RandomForest'] = model
        print("Random Forest regression trained.")

    def train_transformer(self, epochs: int = 100, batch_size: int = 16, learning_rate: float = 1e-3, window_size: int = 24, patience: int = 20):
        """
        Trains a Transformer-based regression model.
    
        Parameters:
        - epochs (int): Number of training epochs.
        - batch_size (int): Batch size for training.
        - learning_rate (float): Learning rate for the optimizer.
        - window_size (int): Number of past observations to use for each prediction.
        - patience (int): Number of epochs to wait for improvement before stopping.
        """
        class TransformerDatasetLocal(Dataset):
            def __init__(self, sequences, targets):
                self.sequences = sequences
                self.targets = targets

            def __len__(self):
                return len(self.sequences)

            def __getitem__(self, idx):
                return self.sequences[idx], self.targets[idx]

        # Scaling the data
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(self.X_train)
        y_scaled = scaler_y.fit_transform(self.y_train.values.reshape(-1, 1)).flatten()

        # Creating sequences
        sequences = []
        targets = []
        for i in range(len(X_scaled) - window_size):
            sequences.append(X_scaled[i:i+window_size])
            targets.append(y_scaled[i+window_size])
        
        sequences = np.array(sequences)
        targets = np.array(targets)

        # Convert to tensors
        sequences = torch.tensor(sequences, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)

        # Create dataset and dataloader
        dataset = TransformerDatasetLocal(sequences, targets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize model, loss, and optimizer
        input_size = self.X_train.shape[1]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = TransformerRegressor(input_size=input_size).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

        # Early stopping parameters
        best_loss = float('inf')
        epochs_no_improve = 0
        early_stop = False

        # Training loop with early stopping
        model.train()
        for epoch in tqdm(range(epochs), desc="Training Transformer", unit="epoch"):
            epoch_losses = []
            
            for seq, target in dataloader:
                seq = seq.to(device)        # Shape: (batch_size, window_size, input_size)
                target = target.to(device)  # Shape: (batch_size)
    
                optimizer.zero_grad()
                output = model(seq)         # Shape: (batch_size, 1)
                loss = criterion(output.squeeze(), target)
                loss.backward()
                optimizer.step()
    
                epoch_losses.append(loss.item())
            
            avg_loss = np.mean(epoch_losses)
            scheduler.step(avg_loss)

            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                epochs_no_improve = 0
                # Save the best model
                best_model_state = model.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {patience} epochs without improvement.")
                    early_stop = True
                    break

            # Print average loss every 20 epochs or the first epoch
            if (epoch+1) % 20 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

            if early_stop:
                break

        # Load the best model
        model.load_state_dict(best_model_state)
        self.models['Transformer'] = {'model': model, 'scaler_X': scaler_X, 'scaler_y': scaler_y, 'window_size': window_size}
        print("Transformer regression trained.")

    def evaluate_models(self):
        """
        Evaluates all trained models and stores the results.
        """
        for name, model in self.models.items():
            if name != 'Transformer':
                predictions = model.predict(self.X_test)
            else:
                # For Transformer, perform scaling and forecasting
                transformer_info = model
                scaler_X = transformer_info['scaler_X']
                scaler_y = transformer_info['scaler_y']
                window_size = transformer_info['window_size']
                transformer_model = transformer_info['model']
                transformer_model.eval()
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                # Prepare test data
                X_test_scaled = scaler_X.transform(self.X_test)
                y_test_scaled = scaler_y.transform(self.y_test.values.reshape(-1, 1)).flatten()

                sequences = []
                targets = []
                for i in range(len(X_test_scaled) - window_size):
                    sequences.append(X_test_scaled[i:i+window_size])
                    targets.append(y_test_scaled[i+window_size])
                
                sequences = np.array(sequences)
                targets = np.array(targets)

                sequences = torch.tensor(sequences, dtype=torch.float32).to(device)
                targets = torch.tensor(targets, dtype=torch.float32).to(device)

                with torch.no_grad():
                    predictions_scaled = transformer_model(sequences).cpu().numpy().flatten()
                
                predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1,1)).flatten()
            
            mse = mean_squared_error(self.y_test, predictions)
            r2 = r2_score(self.y_test, predictions)
            self.results[name] = {'MSE': mse, 'R2': r2}
            print(f"{name} - MSE: {mse:.4f}, R2: {r2:.4f}")

    def plot_results(self):
        """
        Plots actual vs predicted values for all models.
        """
        plt.figure(figsize=(10, 6))
        for name, model in self.models.items():
            if name != 'Transformer':
                predictions = model.predict(self.X_test)
            else:
                # For Transformer, perform scaling and forecasting
                transformer_info = model
                scaler_X = transformer_info['scaler_X']
                scaler_y = transformer_info['scaler_y']
                window_size = transformer_info['window_size']
                transformer_model = transformer_info['model']
                transformer_model.eval()
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                # Prepare test data
                X_test_scaled = scaler_X.transform(self.X_test)
                y_test_scaled = scaler_y.transform(self.y_test.values.reshape(-1, 1)).flatten()

                sequences = []
                targets = []
                for i in range(len(X_test_scaled) - window_size):
                    sequences.append(X_test_scaled[i:i+window_size])
                    targets.append(y_test_scaled[i+window_size])
                
                sequences = np.array(sequences)
                targets = np.array(targets)

                sequences = torch.tensor(sequences, dtype=torch.float32).to(device)
                targets = torch.tensor(targets, dtype=torch.float32).to(device)

                with torch.no_grad():
                    predictions_scaled = transformer_model(sequences).cpu().numpy().flatten()
                
                predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1,1)).flatten()
            
            sns.scatterplot(x=self.y_test, y=predictions, label=name)
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_results(self) -> dict:
        """
        Returns the evaluation results.

        Returns:
        - dict: Evaluation metrics for each model.
        """
        return self.results

# ========================= Survival Analysis =========================

class KaplanMeierEstimator:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the Kaplan-Meier estimator.
        
        Args:
        data (pd.DataFrame): DataFrame containing 'durations' and 'event_observed' columns.
        """
        required_columns = {'durations', 'event_observed'}
        if not required_columns.issubset(data.columns):
            missing = required_columns - set(data.columns)
            raise ValueError(f"The following required columns are missing for Kaplan-Meier Estimation: {missing}")
        
        self.data = data.copy()
        self.km_fit = KaplanMeierFitter()

    def fit(self):
        """Fit the Kaplan-Meier model."""
        self.km_fit.fit(durations=self.data['durations'], event_observed=self.data['event_observed'])

    def plot_survival_curve(self):
        """Plot the Kaplan-Meier survival curve."""
        self.km_fit.plot_survival_function()
        plt.title('Kaplan-Meier Survival Curve')
        plt.xlabel('Time (months)')
        plt.ylabel('Survival Probability')
        plt.grid(True)
        plt.show()

class CoxProportionalHazardsModel:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the Cox proportional hazards model.
        
        Args:
        data (pd.DataFrame): DataFrame containing:
            - 'duration': Time durations (e.g., months) for customer retention.
            - 'event': Event observations (1: churned, 0: not churned).
            - Other covariates (e.g., age, income, purchase frequency).
        """
        required_columns = {'duration', 'event'}
        if not required_columns.issubset(data.columns):
            missing = required_columns - set(data.columns)
            raise ValueError(f"The following required columns are missing for Cox Proportional Hazards Model: {missing}")
        
        self.data = data.copy()
        self.cox_model = CoxPHFitter()

    def fit(self):
        """Fit the Cox proportional hazards model."""
        self.cox_model.fit(self.data, duration_col='duration', event_col='event')
        print("Cox Proportional Hazards Model Summary:")
        print(self.cox_model.summary)

    def plot_survival_curve(self, covariates: dict = None):
        """
        Plot the survival curve.
        
        Args:
        covariates (dict or None): Dictionary of specific covariates to plot.
                                     Example: {'age': 30, 'income': 60000, 'purchase_frequency': 8}.
        """
        plt.figure()
        if covariates:
            # Create a sample for the specified covariates
            sample = pd.DataFrame([covariates])
            survival_function = self.cox_model.predict_survival_function(sample)
            plt.step(survival_function.index, survival_function.iloc[:, 0], where="post", label=str(covariates))
        else:
            # Plot the baseline survival curve
            survival_function = self.cox_model.predict_survival_function(self.data)
            for column in survival_function.columns:
                plt.step(survival_function.index, survival_function[column], where="post", alpha=0.3)
        
        plt.title('Cox Proportional Hazards Model Survival Curve')
        plt.xlabel('Time (months)')
        plt.ylabel('Survival Probability')
        if covariates:
            plt.legend()
        plt.grid(True)
        plt.show()

# ========================= Time Series Forecasting =========================

class TimeSeriesForecaster:
    """
    Time Series Forecasting using ARIMA and Transformer models.
    Automatically selects the appropriate method based on data frequency and size.
    """
    def __init__(self, data: pd.DataFrame, date_column: str = 'Date', sales_column: str = 'Sales', frequency: str = None, cycle_period: int = None):
        """
        Initializes the forecaster with historical data.

        Parameters:
        - data (pd.DataFrame): Historical sales data with 'Date' and 'Sales' columns.
        - date_column (str): Column name for dates.
        - sales_column (str): Column name for sales figures.
        - frequency (str): Frequency of the data (e.g., 'MS' for month start, 'D' for daily, 'W' for weekly).
                           If None, it will be inferred from the data's Date index.
        - cycle_period (int): Number of data points that make up one complete cycle (e.g., 12 for monthly data).
                              If None, it will be set based on the frequency.
        """
        self.data = data.copy()
        self.date_column = date_column
        self.sales_column = sales_column
        self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])
        self.data.set_index(self.date_column, inplace=True)

        # Infer frequency if not provided
        if frequency is None:
            inferred_freq = pd.infer_freq(self.data.index)
            if inferred_freq is None:
                raise ValueError("Cannot infer frequency. Please provide the 'frequency' parameter.")
            self.frequency = inferred_freq
        else:
            self.frequency = frequency

        # Set default cycle_period based on frequency if not provided
        if cycle_period is None:
            if self.frequency.startswith('M'):  # Monthly
                self.cycle_period = 12
            elif self.frequency.startswith('W'):  # Weekly
                self.cycle_period = 52
            elif self.frequency.startswith('D'):  # Daily
                self.cycle_period = 7  # Weekly seasonality
            else:
                raise ValueError(f"Unsupported frequency '{self.frequency}'. Please specify 'cycle_period'.")
        else:
            self.cycle_period = cycle_period

        self.scaler = StandardScaler()
        self.models = {}
        self.predictions = None
        self.future_dates = None

    def determine_method(self) -> str:
        """
        Determines the forecasting method based on the number of complete cycles in the data.

        Returns:
        - method (str): 'ARIMA' or 'Transformer'
        """
        total_cycles = len(self.data) / self.cycle_period
        if 2 <= total_cycles < 4:
            return 'ARIMA'
        elif total_cycles >= 4:
            return 'Transformer'
        else:
            raise ValueError("Insufficient data for forecasting. Require at least two complete cycles.")

    def forecast_arima(self, steps: int = 6) -> pd.Series:
        """
        Forecasts future sales using ARIMA model with seasonal decomposition.

        Parameters:
        - steps (int): Number of future periods to forecast.

        Returns:
        - forecast (pd.Series): Forecasted sales values.
        """
        # Seasonal decomposition
        decomposition = seasonal_decompose(self.data[self.sales_column], period=self.cycle_period, model='additive', extrapolate_trend='freq')
        trend = decomposition.trend
        seasonal = decomposition.seasonal

        # Handle missing values in trend
        trend = trend.fillna(method='bfill').fillna(method='ffill')

        # Fit ARIMA on the trend component
        try:
            # Using (1,1,1) ARIMA for simplicity
            model = ARIMA(trend, order=(1,1,1))
            fitted_model = model.fit()
        except Exception as e:
            print(f"ARIMA model fitting failed: {e}")
            return pd.Series([np.nan]*steps, index=pd.date_range(start=self.data.index[-1] + pd.tseries.frequencies.to_offset(self.frequency), periods=steps, freq=self.frequency))

        # Forecast the trend
        trend_forecast = fitted_model.forecast(steps=steps)

        # Ensure seasonal component is available
        if len(seasonal) < self.cycle_period:
            raise ValueError("Not enough seasonal data to perform forecasting.")

        # Extend seasonal component for forecasting
        seasonal_forecast = []
        last_seasonal = seasonal[-self.cycle_period:]
        for i in range(steps):
            seasonal_forecast.append(last_seasonal.iloc[i % self.cycle_period])

        # Combine trend and seasonal forecasts
        forecast = trend_forecast.values + np.array(seasonal_forecast)

        # Create a Pandas Series for the forecast
        forecast_dates = pd.date_range(start=self.data.index[-1] + pd.tseries.frequencies.to_offset(self.frequency), periods=steps, freq=self.frequency)
        forecast_series = pd.Series(forecast, index=forecast_dates)

        self.predictions = forecast_series
        self.future_dates = forecast_dates
        self.models['ARIMA'] = fitted_model
        return forecast_series

    def prepare_transformer_data(self, window_size: int = 24) -> DataLoader:
        """
        Prepares data for Transformer model by creating sequences.

        Parameters:
        - window_size (int): Number of past observations to use for each prediction.

        Returns:
        - train_loader (DataLoader): DataLoader for training data.
        """
        sales = self.data[self.sales_column].values

        # Scaling the data
        sales_scaled = self.scaler.fit_transform(sales.reshape(-1, 1)).flatten()

        sequences = []
        targets = []
        for i in range(len(sales_scaled) - window_size):
            sequences.append(sales_scaled[i:i+window_size])
            targets.append(sales_scaled[i+window_size])

        sequences = np.array(sequences)
        targets = np.array(targets)

        # Convert to tensors
        sequences = torch.tensor(sequences, dtype=torch.float32).unsqueeze(-1)  # Shape: (num_samples, window_size, 1)
        targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)      # Shape: (num_samples, 1)

        dataset = TimeSeriesDataset(sequences, targets)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)  # Increased batch size for better training

        return train_loader

    def forecast_transformer(self, steps: int = 6, window_size: int = 24, epochs: int = 200, learning_rate: float = 1e-3, patience: int = 20) -> pd.Series:
        """
        Forecasts future sales using Transformer model with improved training.

        Parameters:
        - steps (int): Number of future periods to forecast.
        - window_size (int): Number of past observations to use for each prediction.
        - epochs (int): Number of training epochs.
        - learning_rate (float): Learning rate for the optimizer.
        - patience (int): Number of epochs to wait for improvement before stopping.

        Returns:
        - forecast (pd.Series): Forecasted sales values.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Prepare data
        train_loader = self.prepare_transformer_data(window_size=window_size)

        # Initialize model, loss, and optimizer
        model = TransformerRegressor(input_size=1).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

        # Early stopping parameters
        best_loss = float('inf')
        epochs_no_improve = 0
        early_stop = False

        # Training loop with early stopping
        model.train()
        for epoch in tqdm(range(epochs), desc="Training Transformer", unit="epoch"):
            epoch_losses = []
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)        # Shape: (batch_size, window_size, input_size)
                batch_y = batch_y.to(device)        # Shape: (batch_size, 1)

                optimizer.zero_grad()
                output = model(batch_X)             # Shape: (batch_size, 1)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())
            
            avg_loss = np.mean(epoch_losses)
            scheduler.step(avg_loss)

            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                epochs_no_improve = 0
                # Save the best model
                best_model_state = model.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {patience} epochs without improvement.")
                    early_stop = True
                    break

            # Print average loss every 20 epochs or the first epoch
            if (epoch+1) % 20 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

            if early_stop:
                break

        # Load the best model
        model.load_state_dict(best_model_state)

        # Forecasting
        model.eval()
        sales = self.data[self.sales_column].values
        sales_scaled = self.scaler.transform(sales.reshape(-1, 1)).flatten()
        forecast_scaled = []

        # Initialize the current sequence with the last window_size points
        current_seq = torch.tensor(sales_scaled[-window_size:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)  # Shape: (1, window_size, 1)

        for _ in range(steps):
            with torch.no_grad():
                pred_scaled = model(current_seq).cpu().numpy().flatten()[0]
            forecast_scaled.append(pred_scaled)
            # Update the current sequence by removing the first element and adding the new prediction
            new_input = torch.tensor([[pred_scaled]], dtype=torch.float32).unsqueeze(-1).to(device)  # Shape: (1, 1, 1)
            current_seq = torch.cat((current_seq[:,1:,:], new_input), dim=1)  # Shape: (1, window_size, 1)

        # Inverse scaling
        forecast = self.scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()

        # Generate future dates based on frequency
        last_date = self.data.index[-1]
        freq_offset = pd.tseries.frequencies.to_offset(self.frequency)
        forecast_dates = [last_date + (freq_offset * (i+1)) for i in range(steps)]
        forecast_series = pd.Series(forecast, index=forecast_dates)

        self.predictions = forecast_series
        self.future_dates = forecast_dates
        self.models['Transformer'] = model
        return forecast_series

    def plot_forecast(self, forecast: pd.Series, method: str):
        """
        Plots historical sales data along with the forecasted values.

        Parameters:
        - forecast (pd.Series): Forecasted sales values.
        - method (str): Forecasting method used ('ARIMA' or 'Transformer').
        """
        plt.figure(figsize=(12,6))
        plt.plot(self.data[self.sales_column], label='Historical Sales')
        plt.plot(forecast, label=f'Forecasted Sales ({method})', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.title(f'Sales Forecast using {method}')
        plt.legend()
        plt.grid(True)
        plt.show()

    def run_forecast(self, steps: int = 6, window_size: int = 24, epochs: int = 200, lr: float = 0.001) -> pd.Series:
        """
        Executes the forecasting process by selecting the appropriate method.

        Parameters:
        - steps (int): Number of future periods to forecast.
        - window_size (int): Number of past observations to use for each prediction (used for Transformer).
        - epochs (int): Number of training epochs for Transformer.
        - lr (float): Learning rate for the Transformer optimizer.

        Returns:
        - forecast (pd.Series): Forecasted sales values.
        - method (str): Forecasting method used.
        """
        method = self.determine_method()
        print(f"Selected Forecasting Method: {method}")

        if method == 'ARIMA':
            forecast = self.forecast_arima(steps=steps)
        elif method == 'Transformer':
            forecast = self.forecast_transformer(steps=steps, window_size=window_size, epochs=epochs, learning_rate=lr)
        else:
            raise ValueError("Unsupported forecasting method.")

        self.plot_forecast(forecast, method)
        return forecast

# ========================= Data Analysis Manager =========================

class DataAnalysisManager:
    """
    A manager class to handle various data analyses based on input commands.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the DataAnalysisManager with a DataFrame.

        Parameters:
        - df (pd.DataFrame): The DataFrame containing the data to analyze.
        """
        self.df = df.copy()
        self.clustering = None
        self.correlation = None
        self.regression = None
        self.kaplan_meier = None
        self.cox_model = None
        self.time_series_forecaster = None

    def perform_analysis(self, analysis_command: dict):
        """
        Performs the analysis based on the provided command.

        Parameters:
        - analysis_command (dict): A dictionary specifying the analysis method and parameters.
                                   Example: {"method": "Clustering_DBSCAN", "features": ["Advertising", "Sales"], "eps": 0.5, "min_samples": 2}

        Returns:
        - result: The result of the analysis, which could be a DataFrame, plot, or printed output.
        """
        method = analysis_command.get("method", "").lower()
        if not method:
            raise ValueError("No method specified in the analysis command.")

        if method.startswith("correlation"):
            # Example: "Correlation_Spearman"
            try:
                _, corr_type = method.split("_")
            except ValueError:
                raise ValueError("Correlation method should be in the format 'Correlation_Type' (e.g., 'Correlation_Spearman').")
            predictors = analysis_command.get("predictor", [])
            targets = analysis_command.get("target", [])

            if not predictors or not targets:
                raise ValueError("Predictors and targets must be specified for correlation analysis.")

            # Subset the DataFrame
            subset_df = self.df[predictors + targets]

            # Initialize CorrelationAnalyzer
            self.correlation = CorrelationAnalyzer(subset_df)

            # Calculate specified correlation
            if corr_type.lower() == "pearson":
                self.correlation.calculate_pearson()
                corr_matrix = self.correlation.pearson_corr
                print("Pearson Correlation Matrix:")
                print(corr_matrix)
                self.correlation.visualize_correlation(method='pearson')
            elif corr_type.lower() == "spearman":
                self.correlation.calculate_spearman()
                corr_matrix = self.correlation.spearman_corr
                print("Spearman Correlation Matrix:")
                print(corr_matrix)
                self.correlation.visualize_correlation(method='spearman')
            elif corr_type.lower() == "kendall":
                self.correlation.calculate_kendall()
                corr_matrix = self.correlation.kendall_corr
                print("Kendall Correlation Matrix:")
                print(corr_matrix)
                self.correlation.visualize_correlation(method='kendall')
            else:
                raise ValueError("Unsupported correlation type. Choose 'Pearson', 'Spearman', or 'Kendall'.")

            # Optionally, return pairwise correlations with p-values
            pairwise_corr = self.correlation.pairwise_correlation(method=corr_type.lower())
            print("\nPairwise Correlations with P-values:")
            print(pairwise_corr)
            return pairwise_corr

        elif method.startswith("clustering"):
            # Example: "Clustering_DBSCAN"
            try:
                _, cluster_type = method.split("_")
            except ValueError:
                raise ValueError("Clustering method should be in the format 'Clustering_Type' (e.g., 'Clustering_DBSCAN').")
            feature_columns = analysis_command.get("features", [])
            if not feature_columns:
                raise ValueError("Feature columns must be specified for clustering analysis.")

            # Initialize ClusteringAnalysis
            self.clustering = ClusteringAnalysis(self.df, feature_columns)

            # Perform specified clustering
            if cluster_type.lower() == "kmeans":
                n_clusters = analysis_command.get("n_clusters", 3)
                self.clustering.kmeans_cluster(n_clusters=n_clusters)
            elif cluster_type.lower() == "dbscan":
                eps = analysis_command.get("eps", 0.5)
                min_samples = analysis_command.get("min_samples", 5)
                self.clustering.dbscan_cluster(eps=eps, min_samples=min_samples)
            elif cluster_type.lower() == "hierarchical":
                n_clusters = analysis_command.get("n_clusters", 3)
                linkage = analysis_command.get("linkage", 'ward')
                self.clustering.hierarchical_cluster(n_clusters=n_clusters, linkage=linkage)
            else:
                raise ValueError("Unsupported clustering type. Choose 'KMeans', 'DBSCAN', or 'Hierarchical'.")

            print(f"{cluster_type} clustering performed.")
            return self.clustering.df

        elif method.startswith("regression"):
            # Example: "Regression_Ridge"
            try:
                _, regression_type = method.split("_")
            except ValueError:
                raise ValueError("Regression method should be in the format 'Regression_Type' (e.g., 'Regression_Ridge').")
            predictors = analysis_command.get("predictor", [])
            target = analysis_command.get("target", None)

            if not predictors or not target:
                raise ValueError("Predictors and target must be specified for regression analysis.")

            # Initialize RegressionAnalysis
            self.regression = RegressionAnalysis(self.df, target=target[0])
            self.regression.set_features(predictors)
            self.regression.train_test_split_data()

            # Train specified regression model
            if regression_type.lower() == "ridge":
                alpha = analysis_command.get("alpha", 1.0)
                self.regression.train_ridge(alpha=alpha)
            elif regression_type.lower() == "lasso":
                alpha = analysis_command.get("alpha", 0.1)
                self.regression.train_lasso(alpha=alpha)
            elif regression_type.lower() == "svm":
                C = analysis_command.get("C", 1.0)
                kernel = analysis_command.get("kernel", 'rbf')
                self.regression.train_svm(C=C, kernel=kernel)
            elif regression_type.lower() == "randomforest":
                n_estimators = analysis_command.get("n_estimators", 100)
                random_state = analysis_command.get("random_state", 42)
                self.regression.train_random_forest(n_estimators=n_estimators, random_state=random_state)
            elif regression_type.lower() == "transformer":
                epochs = analysis_command.get("epochs", 200)
                batch_size = analysis_command.get("batch_size", 16)
                lr = analysis_command.get("lr", 1e-3)
                window_size = analysis_command.get("window_size", 24)
                patience = analysis_command.get("patience", 20)
                self.regression.train_transformer(epochs=epochs, batch_size=batch_size, learning_rate=lr, window_size=window_size, patience=patience)
            else:
                raise ValueError("Unsupported regression type. Choose 'Ridge', 'Lasso', 'SVM', 'RandomForest', or 'Transformer'.")

            # Evaluate models
            self.regression.evaluate_models()
            results = self.regression.get_results()
            print("\nRegression Model Evaluation:")
            for model_name, metrics in results.items():
                print(f"{model_name}: MSE = {metrics['MSE']:.4f}, R2 = {metrics['R2']:.4f}")

            # Plot results
            self.regression.plot_results()
            return results

        elif method == "kaplan_meier":
            # Example: {"method": "Kaplan_Meier", "data": pd.DataFrame with 'durations' and 'event_observed'}
            data = analysis_command.get("data", None)

            if not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be provided as a pd.DataFrame for Kaplan-Meier analysis.")

            if not {'durations', 'event_observed'}.issubset(data.columns):
                missing = {'durations', 'event_observed'} - set(data.columns)
                raise ValueError(f"The following required columns are missing for Kaplan-Meier Estimation: {missing}")

            # Initialize KaplanMeierEstimator
            self.kaplan_meier = KaplanMeierEstimator(data=data)
            self.kaplan_meier.fit()
            self.kaplan_meier.plot_survival_curve()
            return self.kaplan_meier.km_fit

        elif method == "cox_proportional_hazards":
            # Example: {"method": "Cox_Proportional_Hazards", "data": pd.DataFrame with 'duration', 'event', and covariates}
            data = analysis_command.get("data", None)
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be provided as a pd.DataFrame for Cox Proportional Hazards analysis.")

            # Initialize CoxProportionalHazardsModel
            self.cox_model = CoxProportionalHazardsModel(data=data)
            self.cox_model.fit()
            self.cox_model.plot_survival_curve()
            return self.cox_model.cox_model

        elif method == "time_series_forecast":
            # Example: {"method": "Time_Series_Forecast", "steps": 6, "window_size": 24, "epochs": 200, "lr": 0.001}
            steps = analysis_command.get("steps", 6)
            window_size = analysis_command.get("window_size", 24)
            epochs = analysis_command.get("epochs", 200)
            lr = analysis_command.get("lr", 0.001)
            cycle_period = analysis_command.get("cycle_period", None)

            sales_column = analysis_command.get("sales_column", "Sales")
            date_column = analysis_command.get("date_column", "Date")
            frequency = analysis_command.get("frequency", None)

            # Initialize TimeSeriesForecaster
            self.time_series_forecaster = TimeSeriesForecaster(
                data=self.df,
                date_column=date_column,
                sales_column=sales_column,
                frequency=frequency,
                cycle_period=cycle_period
            )

            # Run forecast
            forecast = self.time_series_forecaster.run_forecast(
                steps=steps,
                window_size=window_size,
                epochs=epochs,
                lr=lr
            )
            return forecast

        else:
            raise ValueError(f"Unsupported analysis method: {method}")

# ========================= Example Usages =========================

if __name__ == "__main__":
    # ------------------- Example 1: Correlation Analysis (Spearman) -------------------
    print("===== Example 1: Spearman Correlation Analysis =====")
    # Sample DataFrame for Correlation Analysis
    correlation_data = pd.DataFrame({
        'Advertising': [100, 200, 300, 400, 500],
        'Sales': [10, 20, 30, 40, 50],
        'Foot_Traffic': [300, 400, 500, 600, 700],
        'Promotions': [5, 10, 15, 20, 25]
    })

    # Initialize DataAnalysisManager with the DataFrame
    manager_corr = DataAnalysisManager(correlation_data)

    # Define the analysis command for Spearman Correlation
    analysis_command_corr = {
        "method": "Correlation_Spearman",
        "predictor": ["Advertising"],
        "target": ["Sales"]
    }

    # Perform the correlation analysis
    result_corr = manager_corr.perform_analysis(analysis_command_corr)

    # Display the pairwise correlations
    print("\nPairwise Correlations:")
    print(result_corr)

    # ------------------- Example 2: Clustering Analysis (KMeans) -------------------
    print("\n===== Example 2: KMeans Clustering =====")
    # Sample DataFrame for Clustering Analysis
    clustering_data = pd.DataFrame({
        'customer_id': [1, 2, 3, 4, 5, 6],
        'Advertising': [100, 200, 300, 400, 500, 600],
        'Sales': [10, 20, 30, 40, 50, 60],
        'Foot_Traffic': [300, 400, 500, 600, 700, 800],
        'Promotions': [5, 10, 15, 20, 25, 30]
    })

    # Initialize DataAnalysisManager with the DataFrame
    manager_cluster = DataAnalysisManager(clustering_data)

    # Define the analysis command for KMeans Clustering
    analysis_command_cluster = {
        "method": "Clustering_KMeans",
        "features": ["Advertising", "Sales"],
        "n_clusters": 2
    }

    # Perform the clustering analysis
    clustered_data = manager_cluster.perform_analysis(analysis_command_cluster)

    # Display the clustered DataFrame
    print("\nClustered Data:")
    print(clustered_data)

    # ------------------- Example 3: Regression Analysis (Ridge) -------------------
    print("\n===== Example 3: Ridge Regression =====")
    # Sample DataFrame for Regression Analysis
    regression_data = pd.DataFrame({
        'Advertising': [100, 200, 300, 400, 500, 600],
        'Foot_Traffic': [300, 400, 500, 600, 700, 800],
        'Sales': [10, 20, 30, 40, 50, 60]
    })

    # Initialize DataAnalysisManager with the DataFrame
    manager_reg = DataAnalysisManager(regression_data)

    # Define the analysis command for Ridge Regression
    analysis_command_reg = {
        "method": "Regression_Ridge",
        "predictor": ["Advertising", "Foot_Traffic"],
        "target": ["Sales"],
        "alpha": 1.0
    }

    # Perform the regression analysis
    results_reg = manager_reg.perform_analysis(analysis_command_reg)

    # Display the evaluation results
    print("\nRegression Model Evaluation:")
    print(results_reg)

    # ------------------- Example 4: Survival Analysis (Kaplan-Meier) -------------------
    print("\n===== Example 4: Kaplan-Meier Estimation =====")
    # Sample Data for Kaplan-Meier Estimation
    km_data = pd.DataFrame({
        'durations': [5, 8, 12, 7, 9, 6, 15, 10, 4, 3, 11, 14, 18, 20, 25],  # Retention time in months
        'event_observed': [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1]  # 1: churned, 0: not churned
    })

    # Initialize DataAnalysisManager with an empty DataFrame (Kaplan-Meier uses separate data)
    manager_km = DataAnalysisManager(pd.DataFrame())

    # Define the analysis command for Kaplan-Meier
    analysis_command_km = {
        "method": "Kaplan_Meier",
        "data": km_data
    }

    # Perform the Kaplan-Meier analysis
    km_fit = manager_km.perform_analysis(analysis_command_km)

    # ------------------- Example 5: Survival Analysis (Cox Proportional Hazards Model) -------------------
    print("\n===== Example 5: Cox Proportional Hazards Model =====")
    # Sample Data for Cox Proportional Hazards Model
    cox_data = pd.DataFrame({
        'duration': [5, 8, 12, 7, 9, 6, 15, 10, 4, 3, 11, 14, 18, 20, 25],  # Retention time in months
        'event': [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1],         # 1: churned, 0: not churned
        'age': [25, 34, 45, 23, 30, 37, 50, 29, 40, 22, 28, 33, 47, 55, 60],          # Customer age
        'income': [50000, 65000, 75000, 55000, 60000, 72000, 80000, 68000, 48000, 45000, 70000, 62000, 85000, 90000, 95000],  # Annual income
        'purchase_frequency': [12, 15, 18, 10, 14, 20, 22, 16, 8, 5, 13, 17, 25, 28, 30]  # Annual purchase frequency
    })

    # Initialize DataAnalysisManager with an empty DataFrame (Cox model uses separate data)
    manager_cox = DataAnalysisManager(pd.DataFrame())

    # Define the analysis command for Cox Proportional Hazards
    analysis_command_cox = {
        "method": "Cox_Proportional_Hazards",
        "data": cox_data
    }

    # Perform the Cox Proportional Hazards analysis
    cox_fit = manager_cox.perform_analysis(analysis_command_cox)

    # ------------------- Example 6: Time Series Forecasting (ARIMA) -------------------
    print("\n===== Example 6: ARIMA Time Series Forecasting =====")
    # Sample Monthly Sales Data for ARIMA Forecasting
    arima_data = pd.DataFrame({
        "Sales": [500, 600, 550, 650, 780, 810, 890, 920, 1000, 1100, 1050, 1150,
                  1200, 1300, 1250, 1350, 1400, 1500, 1450, 1550, 1600, 1700, 1650, 1750,
                  1800, 1900, 1850, 1950, 2000, 2100, 2050, 2150, 2200, 2300, 2250, 2350],
        "Date": pd.date_range(start="2021-01-01", periods=36, freq='MS')  # 'MS' for Month Start
    })

    # Initialize DataAnalysisManager with the ARIMA DataFrame
    manager_ts = DataAnalysisManager(arima_data)

    # Define the analysis command for ARIMA Forecasting
    analysis_command_ts = {
        "method": "Time_Series_Forecast",
        "steps": 12,            # Forecasting next 12 months
        "window_size": 24,      # Not used for ARIMA
        "epochs": 200,          # Not used for ARIMA
        "lr": 0.001,            # Not used for ARIMA
        "sales_column": "Sales",
        "date_column": "Date",
        "frequency": "MS"
    }

    # Perform the forecasting analysis
    forecast_ts = manager_ts.perform_analysis(analysis_command_ts)

    # Display the forecasted sales
    print("\nForecasted Sales (ARIMA):")
    print(forecast_ts)

    # ------------------- Example 7: Regression Analysis (Random Forest) -------------------
    print("\n===== Example 7: Random Forest Regression =====")
    # Sample DataFrame for Regression Analysis
    regression_data_rf = pd.DataFrame({
        'Advertising': [100, 200, 300, 400, 500, 600],
        'Foot_Traffic': [300, 400, 500, 600, 700, 800],
        'Sales': [10, 20, 30, 40, 50, 60]
    })

    # Initialize DataAnalysisManager with the DataFrame
    manager_rf = DataAnalysisManager(regression_data_rf)

    # Define the analysis command for Random Forest Regression
    analysis_command_rf = {
        "method": "Regression_RandomForest",
        "predictor": ["Advertising", "Foot_Traffic"],
        "target": ["Sales"],
        "n_estimators": 100,
        "random_state": 42
    }

    # Perform the regression analysis
    results_rf = manager_rf.perform_analysis(analysis_command_rf)

    # Display the evaluation results
    print("\nRandom Forest Regression Evaluation:")
    print(results_rf)

    # ------------------- Example 8: Regression Analysis (Lasso) -------------------
    print("\n===== Example 8: Lasso Regression =====")
    # Sample DataFrame for Regression Analysis
    regression_data_lasso = pd.DataFrame({
        'Advertising': [100, 200, 300, 400, 500, 600],
        'Foot_Traffic': [300, 400, 500, 600, 700, 800],
        'Sales': [10, 20, 30, 40, 50, 60]
    })

    # Initialize DataAnalysisManager with the DataFrame
    manager_lasso = DataAnalysisManager(regression_data_lasso)

    # Define the analysis command for Lasso Regression
    analysis_command_lasso = {
        "method": "Regression_Lasso",
        "predictor": ["Advertising", "Foot_Traffic"],
        "target": ["Sales"],
        "alpha": 0.1
    }

    # Perform the regression analysis
    results_lasso = manager_lasso.perform_analysis(analysis_command_lasso)

    # Display the evaluation results
    print("\nLasso Regression Evaluation:")
    print(results_lasso)

    # ------------------- Example 9: Time Series Forecasting (Transformer) -------------------
    print("\n===== Example 9: Transformer Time Series Forecasting =====")
    # Generate Sample Daily Sales Data with Weekly Seasonality and Trend
    days = 365
    base = 500
    trend_per_day = 0.5
    seasonal_period = 7  # Weekly seasonality
    amplitude = 50

    np.random.seed(42)  # For reproducibility
    noise = np.random.normal(0, 20, days)  # Less noise for better Transformer performance
    trend = np.linspace(0, trend_per_day * days, days)
    seasonal = amplitude * np.sin(2 * np.pi * np.arange(days) / seasonal_period)
    fluctuating_sales = base + trend + seasonal + noise

    # Creating the DataFrame with fluctuating sales data
    transformer_data = pd.DataFrame({
        "Sales": fluctuating_sales.round(2),  # Rounded to 2 decimal places
        "Date": pd.date_range(start="2021-01-01", periods=days, freq='D')  # 'D' for Daily
    })

    # Initialize DataAnalysisManager with the Transformer DataFrame
    manager_transformer = DataAnalysisManager(transformer_data)

    # Define the analysis command for Transformer Forecasting
    analysis_command_transformer = {
        "method": "Time_Series_Forecast",
        "steps": 30,               # Forecasting next 30 days
        "window_size": 30,         # Using the past 30 days to predict the next day
        "epochs": 200,             # Number of training epochs
        "lr": 0.001,               # Learning rate
        "sales_column": "Sales",
        "date_column": "Date",
        "frequency": "D"
    }

    # Perform the forecasting analysis
    forecast_transformer = manager_transformer.perform_analysis(analysis_command_transformer)

    # Display the forecasted sales
    print("\nForecasted Sales (Transformer):")
    print(forecast_transformer)

    # ------------------- Example 10: Clustering Analysis (DBSCAN) with Highlighted Customers -------------------
    print("\n===== Example 10: DBSCAN Clustering with Highlighted Customers =====")
    # Sample DataFrame for Clustering Analysis
    clustering_data_dbscan = pd.DataFrame({
        'customer_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Advertising': [100, 150, 200, 250, 300, 350, 400, 450, 500, 550],
        'Sales': [10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
        'Foot_Traffic': [300, 350, 400, 450, 500, 550, 600, 650, 700, 750],
        'Promotions': [5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    })

    # Initialize DataAnalysisManager with the DBSCAN DataFrame
    manager_dbscan = DataAnalysisManager(clustering_data_dbscan)

    # Define the analysis command for DBSCAN Clustering
    analysis_command_dbscan = {
        "method": "Clustering_DBSCAN",
        "features": ["Advertising", "Sales"],
        "eps": 0.5,
        "min_samples": 2
    }

    # Perform the clustering analysis
    clustered_data_dbscan = manager_dbscan.perform_analysis(analysis_command_dbscan)

    # Define customer IDs to highlight
    highlight_customers = [2, 5, 8]

    # Plot clusters with highlighted customers
    manager_dbscan.clustering.plot_clusters('cluster_dbscan', title='DBSCAN Clustering with Highlights', highlight_ids=highlight_customers)

    # Display the clustered DataFrame
    print("\nDBSCAN Clustered Data:")
    print(clustered_data_dbscan)

    # ------------------- Example 11: Multiple Analysis Methods (Correlation, Clustering, Regression) -------------------
    print("\n===== Example 11: Multiple Analysis Methods (Correlation, Clustering, Regression) =====")
    # Combined Sample DataFrame
    combined_data = pd.DataFrame({
        'Advertising': [100, 200, 300, 400, 500, 600],
        'Sales': [10, 20, 30, 40, 50, 60],
        'Foot_Traffic': [300, 400, 500, 600, 700, 800],
        'Promotions': [5, 10, 15, 20, 25, 30],
        'Age': [25, 34, 45, 23, 30, 37],
        'Income': [50000, 65000, 75000, 55000, 60000, 72000],
        'Purchase_Frequency': [12, 15, 18, 10, 14, 20]
    })

    # Initialize DataAnalysisManager with the Combined DataFrame
    manager_multiple = DataAnalysisManager(combined_data)

    # 1. Spearman Correlation Analysis
    correlation_command_multiple = {
        "method": "Correlation_Spearman",
        "predictor": ["Advertising", "Foot_Traffic"],
        "target": ["Sales", "Promotions"]
    }
    correlation_result_multiple = manager_multiple.perform_analysis(correlation_command_multiple)

    # 2. KMeans Clustering
    clustering_command_multiple = {
        "method": "Clustering_KMeans",
        "features": ["Advertising", "Sales"],
        "n_clusters": 2
    }
    clustered_data_multiple = manager_multiple.perform_analysis(clustering_command_multiple)

    # 3. Ridge Regression
    regression_command_multiple = {
        "method": "Regression_Ridge",
        "predictor": ["Advertising", "Foot_Traffic"],
        "target": ["Sales"],
        "alpha": 1.0
    }
    regression_results_multiple = manager_multiple.perform_analysis(regression_command_multiple)

    # Display all results
    print("\nAll Analysis Results:")
    print("Correlation Analysis Result:")
    print(correlation_result_multiple)
    print("\nClustering Analysis Result:")
    print(clustered_data_multiple)
    print("\nRegression Analysis Result:")
    print(regression_results_multiple)
