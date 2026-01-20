"""Data preprocessing and feature engineering module."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')


class DataAnalyzer:
    """Analyzes data and detects column types."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.column_types: Dict[str, str] = {}
        self.categorical_columns: List[str] = []
        self.numerical_columns: List[str] = []
        self.text_columns: List[str] = []
        self.target_column: Optional[str] = None
        self.analyze()

    def analyze(self) -> None:
        """Automatically detect column types."""
        for col in self.df.columns:
            # Skip if all values are missing
            if self.df[col].isna().all():
                self.column_types[col] = 'drop'
                continue
                
            col_dtype = self.df[col].dtype
            unique_count = self.df[col].nunique()
            total_count = len(self.df[col])
            
            # Check if numeric can be parsed
            if col_dtype == 'object':
                # Try to convert to numeric
                try:
                    numeric_values = pd.to_numeric(self.df[col].dropna(), errors='raise')
                    if pd.notna(numeric_values).all():
                        self.column_types[col] = 'numerical'
                        continue
                except (ValueError, TypeError):
                    pass
                
                # It's a string column
                if unique_count <= 10:
                    self.column_types[col] = 'categorical'
                elif unique_count <= 50:
                    self.column_types[col] = 'categorical'  # Label encodeable
                elif unique_count == total_count:
                    self.column_types[col] = 'id'  # Likely an ID column
                else:
                    self.column_types[col] = 'text'
            elif col_dtype in ['int64', 'float64']:
                if unique_count <= 10 and (self.df[col].dropna().min() >= 0 and self.df[col].dropna().max() <= 1):
                    self.column_types[col] = 'binary'
                elif unique_count <= 20:
                    self.column_types[col] = 'categorical'
                else:
                    self.column_types[col] = 'numerical'
            elif str(col_dtype).startswith('category'):
                self.column_types[col] = 'categorical'
            else:
                self.column_types[col] = 'numerical'

        self.numerical_columns = [c for c, t in self.column_types.items() 
                                  if t in ['numerical', 'binary']]
        self.categorical_columns = [c for c, t in self.column_types.items() 
                                    if t == 'categorical']
        self.text_columns = [c for c, t in self.column_types.items() 
                             if t == 'text']

    def infer_target(self, target_column: Optional[str] = None) -> str:
        """Infer target column or use specified one."""
        if target_column:
            self.target_column = target_column
            return target_column

        # Look for common target column names
        target_names = ['target', 'label', 'class', 'y', 'survived', 'price', 'outcome', 'result']
        for name in target_names:
            if name.lower() in [c.lower() for c in self.df.columns]:
                for col in self.df.columns:
                    if col.lower() == name.lower():
                        self.target_column = col
                        return col

        # Heuristic: column with few unique values, likely the target
        for col in reversed(self.df.columns):
            if col in self.column_types and self.column_types[col] in ['categorical', 'binary']:
                if self.df[col].nunique() <= 20:
                    self.target_column = col
                    return col

        # Default to last column that's not an ID
        for col in reversed(self.df.columns):
            if self.column_types.get(col) != 'id':
                self.target_column = col
                return col

        self.target_column = self.df.columns[-1]
        return self.target_column

    def get_feature_columns(self) -> Tuple[List[str], List[str], List[str]]:
        """Get feature columns (exclude target and ID columns)."""
        if not self.target_column:
            self.infer_target()

        numerical = [c for c in self.numerical_columns if c != self.target_column]
        categorical = [c for c in self.categorical_columns if c != self.target_column]
        text = [c for c in self.text_columns if c != self.target_column]
        
        return numerical, categorical, text

    def get_drop_columns(self) -> List[str]:
        """Get columns that should be dropped."""
        return [c for c, t in self.column_types.items() 
                if t == 'drop' or (c != self.target_column and t == 'id')]

    def get_column_summary(self) -> Dict[str, Any]:
        """Get summary of column analysis."""
        return {
            'columns': list(self.df.columns),
            'numerical': self.numerical_columns,
            'categorical': self.categorical_columns,
            'text': self.text_columns,
            'types': self.column_types,
            'target': self.target_column,
            'shape': self.df.shape,
            'missing': {col: int(self.df[col].isna().sum())
                       for col in self.df.columns if self.df[col].isna().sum() > 0}
        }


class DataPreprocessor:
    """Preprocesses data for ML training."""

    def __init__(self, numerical_columns: List[str], categorical_columns: List[str]):
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_names: Optional[List[str]] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}

    def build_preprocessor(self, strategy: str = 'median') -> 'DataPreprocessor':
        """Build preprocessing pipeline."""
        transformers = []

        if self.numerical_columns:
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=strategy)),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numerical_transformer, self.numerical_columns))

        if self.categorical_columns:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
            ])
            transformers.append(('cat', categorical_transformer, self.categorical_columns))

        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )

        return self

    def fit(self, X: pd.DataFrame) -> 'DataPreprocessor':
        """Fit preprocessor on data."""
        if self.preprocessor is None:
            self.build_preprocessor()

        self.preprocessor.fit(X)
        self._get_feature_names(X)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        return self.preprocessor.transform(X)

    def fit_transform(self, X: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Fit and transform data."""
        self.fit(X)
        return self.transform(X), self.feature_names

    def _get_feature_names(self, X: pd.DataFrame) -> None:
        """Extract feature names after transformation."""
        try:
            self.feature_names = []
            if self.numerical_columns:
                self.feature_names.extend(self.numerical_columns)
            if self.categorical_columns:
                cat_encoder = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
                cat_names = list(cat_encoder.get_feature_names_out(self.categorical_columns))
                self.feature_names.extend(cat_names)
        except Exception:
            self.feature_names = list(X.columns)


class AutoMLDataLoader:
    """Main data loading and preprocessing class."""

    def __init__(self, csv_path: Optional[str] = None, dataframe: Optional[pd.DataFrame] = None):
        self.csv_path = csv_path
        self.dataframe: Optional[pd.DataFrame] = None
        self.analyzer: Optional[DataAnalyzer] = None
        self.preprocessor: Optional[DataPreprocessor] = None
        self.target_column: Optional[str] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}

        if csv_path:
            self.load_csv(csv_path)
        elif dataframe is not None:
            self.dataframe = dataframe

    def load_csv(self, csv_path: str, delimiter: str = ',') -> 'AutoMLDataLoader':
        """Load data from CSV file."""
        self.csv_path = csv_path
        self.dataframe = pd.read_csv(csv_path, delimiter=delimiter)
        print(f"Loaded {len(self.dataframe)} rows, {len(self.dataframe.columns)} columns")
        return self

    def analyze(self, target_column: Optional[str] = None) -> 'AutoMLDataLoader':
        """Analyze data and detect types."""
        if self.dataframe is None:
            raise ValueError("No data loaded. Call load_csv() first.")

        self.analyzer = DataAnalyzer(self.dataframe)
        self.target_column = self.analyzer.infer_target(target_column)

        print(f"Target column: {self.target_column}")
        print(f"Numerical features: {len(self.analyzer.numerical_columns)}")
        print(f"Categorical features: {len(self.analyzer.categorical_columns)}")
        print(f"Text/high-cardinality features: {len(self.analyzer.text_columns)}")

        return self

    def _extract_features_from_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract useful features from text columns."""
        df = df.copy()
        
        # Extract title from Name column
        if 'Name' in df.columns:
            df['Name_Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
            df['Name_Title'] = df['Name_Title'].fillna('Unknown')
            title_mapping = {
                'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
                'Dr': 'Professional', 'Rev': 'Professional', 'Col': 'Military',
                'Major': 'Military', 'Capt': 'Military', 'Countess': 'Nobility',
                'Don': 'Nobility', 'Dona': 'Nobility', 'Jonkheer': 'Nobility',
                'Lady': 'Nobility', 'Sir': 'Nobility', 'Mlle': 'Miss', 'Ms': 'Miss',
                'Mme': 'Mrs', 'NaN': 'Unknown'
            }
            df['Name_Title'] = df['Name_Title'].map(title_mapping).fillna('Other')
        
        # Extract deck from Cabin
        if 'Cabin' in df.columns:
            df['Cabin_Deck'] = df['Cabin'].str[0].fillna('Unknown')
        
        # Extract ticket prefix and number
        if 'Ticket' in df.columns:
            df['Ticket_Prefix'] = df['Ticket'].str.extract(r'^([A-Za-z0-9\s]+)?\s*\d*$', expand=False)
            df['Ticket_Prefix'] = df['Ticket_Prefix'].fillna('None').str.strip()
            df['Ticket_Number'] = df['Ticket'].str.extract(r'(\d+)$', expand=False)
            df['Ticket_Number'] = pd.to_numeric(df['Ticket_Number'], errors='coerce').fillna(0)
        
        # Family size from SibSp and Parch
        if 'SibSp' in df.columns and 'Parch' in df.columns:
            df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
            df['Is_Alone'] = (df['Family_Size'] == 1).astype(int)
        
        return df

    def _label_encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Label encode categorical columns with many unique values."""
        df = df.copy()
        high_card_cols = []
        
        for col in df.columns:
            if df[col].dtype == 'object':
                unique_count = df[col].nunique()
                if unique_count > 10 and unique_count < 1000:
                    high_card_cols.append(col)
        
        for col in high_card_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col + '_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    df[col + '_encoded'] = df[col].map(lambda x: x if x in le.classes_ else 'Unknown')
                    df[col + '_encoded'] = le.transform(df[col + '_encoded'].astype(str))
            df = df.drop(columns=[col])
        
        return df

    def preprocess(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Preprocess data and return X, y."""
        if self.analyzer is None:
            raise ValueError("Data not analyzed. Call analyze() first.")

        numerical, categorical, text = self.analyzer.get_feature_columns()
        drop_cols = self.analyzer.get_drop_columns()
        
        df = self.dataframe.copy()
        
        # Drop columns that must be dropped
        for col in drop_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # Extract features from text columns
        df = self._extract_features_from_text(df)
        
        # Update column lists after feature extraction
        if 'Name_Title' in df.columns:
            categorical.append('Name_Title')
        if 'Cabin_Deck' in df.columns:
            categorical.append('Cabin_Deck')
        if 'Ticket_Prefix' in df.columns:
            categorical.append('Ticket_Prefix')
        if 'Ticket_Number' in df.columns and 'Ticket_Number' not in numerical:
            numerical.append('Ticket_Number')
        if 'Family_Size' in df.columns:
            numerical.append('Family_Size')
        if 'Is_Alone' in df.columns:
            numerical.append('Is_Alone')
        
        # Remove original columns that were processed
        cols_to_remove = ['Name', 'Cabin', 'Ticket']
        for col in cols_to_remove:
            if col in df.columns:
                df = df.drop(columns=[col])
                if col in numerical:
                    numerical.remove(col)
                if col in categorical:
                    categorical.remove(col)
        
        # Remove target from features
        if self.target_column in numerical:
            numerical.remove(self.target_column)
        if self.target_column in categorical:
            categorical.remove(self.target_column)
        
        # Label encode remaining high-cardinality categoricals
        df = self._label_encode_categoricals(df, fit=True)
        categorical = [c for c in categorical if c in df.columns and df[c].dtype != 'object']
        
        # Update numerical columns list
        numerical = [c for c in numerical if c in df.columns]
        
        # Separate features and target
        X = df.drop(columns=[self.target_column]) if self.target_column in df.columns else df
        y = df[self.target_column].values if self.target_column in df.columns else np.array([])
        
        # Handle target encoding if categorical
        if self.analyzer.column_types.get(self.target_column) == 'categorical' or \
           (self.target_column in df.columns and df[self.target_column].dtype == 'object'):
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
        
        # Final column lists
        final_numerical = [c for c in numerical if c in X.columns]
        final_categorical = [c for c in categorical if c in X.columns]
        
        self.preprocessor = DataPreprocessor(final_numerical, final_categorical)
        X_array, feature_names = self.preprocessor.fit_transform(X)

        return X_array, y, feature_names

    def get_summary(self) -> Dict[str, Any]:
        """Get data summary."""
        if self.analyzer is None:
            return {'error': 'Data not analyzed'}
        return self.analyzer.get_column_summary()

    def select_features(self, X: np.ndarray, y: np.ndarray,
                       k: Optional[int] = None, method: str = 'f_classif') -> Tuple[np.ndarray, List[int]]:
        """Select top k features."""
        n_features = X.shape[1]
        k = min(k or n_features, n_features)

        if method == 'f_classif':
            selector = SelectKBest(f_classif, k=k)
        else:
            selector = SelectKBest(mutual_info_classif, k=k)

        X_selected = selector.fit_transform(X, y)
        selected_indices = selector.get_support(indices=True)

        return X_selected, list(selected_indices)


def load_data(csv_path: str, target_column: Optional[str] = None,
              select_features: bool = False, k: int = 10) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load and preprocess data from CSV.

    Args:
        csv_path: Path to CSV file
        target_column: Name of target column (auto-detected if None)
        select_features: Whether to select top features
        k: Number of top features to select

    Returns:
        X: Preprocessed features
        y: Target variable
        info: Dictionary with data info
    """
    loader = AutoMLDataLoader(csv_path)
    loader.load_csv(csv_path)
    loader.analyze(target_column)
    X, y, feature_names = loader.preprocess()

    if select_features and X.shape[1] > k:
        X, indices = loader.select_features(X, y, k=k)
        feature_names = [feature_names[i] for i in indices]

    info = {
        'csv_path': csv_path,
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'feature_names': feature_names,
        'target_column': loader.target_column,
        'target_classes': list(np.unique(y)) if len(np.unique(y)) <= 20 else 'regression',
        'summary': loader.get_summary()
    }

    return X, y, info
