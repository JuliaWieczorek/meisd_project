import pandas as pd
from project.data_preparation import EmotionDataProcessor

class DataQualityChecker:
    def __init__(self, original_data, processed_data):
        """
        Initializes the DataQualityChecker with original and processed datasets.

        Args:
            original_data (pd.DataFrame): The original dataset before processing.
            processed_data (pd.DataFrame): The dataset after processing.
        """
        self.original_data = original_data
        self.processed_data = processed_data

    def check_missing_values(self, df):
        """
        Checks for missing values in the dataset.

        Args:
            df (pd.DataFrame): The dataset to check.

        Returns:
            pd.Series: Count of missing values per column.
        """
        missing_values = df.isna().sum()
        return missing_values[missing_values > 0]

    def check_missing_emotion(self, df):
        emotion_columns = ['emotion', 'emotion2', 'emotion3']

        for col in emotion_columns:
            unmapped = df[f'target_{col}'].isna().sum()
            if unmapped > 0:
                print(f"Warning: {unmapped} emotion values in column {col} could not be mapped.")

                # Fill missing values with a default value (e.g., 0) or another chosen value
                df[f'target_{col}'].fillna(0, inplace=True)
                print(f"Missing emotion values in {col} replaced with default (0).")

    def check_missing_sentiment(self, df, sentiment_column='sentiment'):
        # Check for unmapped values and print a warning if needed
        unmapped = df[sentiment_column].isna().sum()
        if unmapped > 0:
            print(f"Warning: {unmapped} sentiment values could not be mapped.")
            print(f"Unmapped sentiment values: {df[sentiment_column].isna().sum()}")

            # Optionally, fill missing values with 0 or another default value
            df[sentiment_column].fillna(0, inplace=True)
            print(f"Missing sentiment values replaced with default (0).")


    def check_column_types(self, df, expected_types):
        """
        Checks if columns have the expected data types.

        Args:
            df (pd.DataFrame): The dataset to check.
            expected_types (dict): Dictionary of expected data types (e.g., {'emotion': int}).

        Returns:
            dict: Dictionary of columns with mismatched data types.
        """
        mismatched_types = {}
        for col, expected_type in expected_types.items():
            if col in df.columns and not pd.api.types.is_dtype_equal(df[col].dtype, expected_type):
                mismatched_types[col] = df[col].dtype
        return mismatched_types

    def compare_value_counts(self, original_column, processed_column):
        """
        Compares value counts between the original and processed columns.

        Args:
            original_column (pd.Series): The original column.
            processed_column (pd.Series): The processed column.

        Returns:
            pd.DataFrame: DataFrame showing original and processed value counts.
        """
        original_counts = original_column.value_counts()
        processed_counts = processed_column.value_counts()
        comparison = pd.DataFrame({'Original': original_counts, 'Processed': processed_counts}).fillna(0)
        comparison['Difference'] = comparison['Processed'] - comparison['Original']
        return comparison

    def analyze(self, expected_types):
        """
        Performs a full analysis of data quality, including missing values, data types,
        and comparison of values between original and processed datasets.

        Args:
            expected_types (dict): Expected data types for columns in the processed dataset.

        Returns:
            dict: Dictionary containing the analysis results.
        """
        analysis_results = {}

        # Check missing values
        analysis_results['Missing Values (Original)'] = self.check_missing_values(self.original_data)
        analysis_results['Missing Values (Processed)'] = self.check_missing_values(self.processed_data)

        # Check data types
        analysis_results['Mismatched Data Types'] = self.check_column_types(self.processed_data, expected_types)

        # Compare value counts for key columns
        comparison_results = {}
        for col in expected_types.keys():
            if col in self.original_data.columns and col in self.processed_data.columns:
                comparison_results[col] = self.compare_value_counts(self.original_data[col], self.processed_data[col])
        analysis_results['Value Count Comparison'] = comparison_results

        return analysis_results

    def print_analysis(self, analysis_results):
        """
        Prints the analysis results in a readable format.

        Args:
            analysis_results (dict): The dictionary containing analysis results.
        """
        print("\n--- Data Quality Analysis ---\n")

        # Missing Values
        print("Missing Values (Original):")
        print(analysis_results['Missing Values (Original)'], "\n")

        print("Missing Values (Processed):")
        print(analysis_results['Missing Values (Processed)'], "\n")

        # Mismatched Data Types
        print("Mismatched Data Types:")
        for col, dtype in analysis_results['Mismatched Data Types'].items():
            print(f"  - Column '{col}' has type '{dtype}', expected '{expected_types[col]}'")
        print()

        # Value Count Comparison
        print("Value Count Comparison:")
        for col, comparison in analysis_results['Value Count Comparison'].items():
            print(f"\nColumn: {col}")
            print(comparison)

