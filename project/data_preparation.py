import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

class EmotionDataProcessor:
    def __init__(self, emotion_map, test_size=0.3, random_state=42):
        """
        Initializes the EmotionDataProcessor with configuration.

        Args:
            emotion_map (dict): Mapping of emotions to numerical values.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Random seed for reproducibility.
        """
        self.emotion_map = emotion_map
        self.test_size = test_size
        self.random_state = random_state
        self.mlb = MultiLabelBinarizer()

    def load_data(self, file_path):
        """
        Loads data from a CSV file.
        """
        return pd.read_csv(file_path)

    def get_unique_emotions(self, df):
        """
        Identifies all unique emotions in the dataset.

        Args:
            df (DataFrame): The dataset.

        Returns:
            set: A set of unique emotions across all emotion columns.
        """
        emotion_cols = ['emotion', 'emotion2', 'emotion3']
        unique_emotions = set()
        for col in emotion_cols:
            unique_emotions.update(df[col].dropna().unique())
        return unique_emotions

    def generate_emotion_map(self, unique_emotions):
        """
        Generates a mapping of emotions to numerical values.

        Args:
            unique_emotions (set): A set of unique emotions.

        Returns:
            dict: A dictionary mapping emotions to integers.
        """
        return {emotion: idx for idx, emotion in enumerate(sorted(unique_emotions))}

    def map_emotions(self, df):
        """
        Maps emotion columns to numerical values using the emotion_map.

        Args:
            df (DataFrame): The dataset.

        Returns:
            DataFrame: The dataset with mapped emotion columns.
        """
        if self.emotion_map is None:
            unique_emotions = self.get_unique_emotions(df)
            self.emotion_map = self.generate_emotion_map(unique_emotions)
        for col in ['emotion', 'emotion2', 'emotion3']:
            df[f'target_{col}'] = df[col].map(self.emotion_map)
        return df

    def count_emotion_occurrences(self, df):
        """
        Counts the occurrences of each emotion in the dataset.

        Args:
            df (DataFrame): The dataset.

        Returns:
            dict: A dictionary with emotion counts.
        """
        emotion_cols = ['emotion', 'emotion2', 'emotion3']
        all_emotions = df[emotion_cols].values.flatten()
        all_emotions = [e for e in all_emotions if pd.notna(e)]
        return pd.Series(all_emotions).value_counts().to_dict()

    def fill_missing_utterances(self, df):
        """
        Fills missing values in the 'Utterances' column with 'missing'.

        Args:
            df (DataFrame): The dataset.

        Returns:
            DataFrame: The dataset with filled 'Utterances' column.
        """
        df['Utterances'] = df['Utterances'].fillna("missing")
        return df


    # def map_emotions(self, df):
    #     """
    #     Maps emotion columns to numerical values.
    #     """
    #     for col in ['emotion', 'emotion2', 'emotion3']:
    #         df[f'target_{col}'] = df[col].map(self.emotion_map)
    #     return df

    def fill_missing_emotions(self, df):
        """
        Fills missing values in emotion columns using forward and backward fill.
        """
        emotion_cols = [col for col in df.columns if col.startswith('target_')]
        df[emotion_cols] = df[emotion_cols].apply(lambda row: row.bfill().ffill(), axis=1)
        return df

    def combine_emotions(self, df):
        """
        Combines all unique non-NaN emotion values into a single list.
        """
        emotion_cols = [col for col in df.columns if col.startswith('target_')]
        df['combined_emotions'] = df[emotion_cols].apply(lambda x: x.dropna().unique().tolist(), axis=1)
        return df

    def binarize_emotions(self, df):
        """
        Converts combined emotions into binary format using MultiLabelBinarizer.
        """
        emotion_binarized = self.mlb.fit_transform(df['combined_emotions'])
        emotion_df = pd.DataFrame(emotion_binarized, columns=[f'emotion_{i}' for i in range(emotion_binarized.shape[1])])
        return pd.concat([df['Utterances'], emotion_df], axis=1)

    def split_data(self, df):
        """
        Splits the dataset into train, validation, and test sets.
        """
        train_df, temp_df = train_test_split(df, test_size=self.test_size, random_state=self.random_state)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=self.random_state)
        return train_df, val_df, test_df

    def process(self, file_path):
        """
        Main method to process data:
        - Load data
        - Map and process emotions
        - Combine and binarize emotion labels
        - Split into train, validation, and test sets
        """
        df = self.load_data(file_path)
        df = self.map_emotions(df)
        df = self.fill_missing_emotions(df)
        df = self.combine_emotions(df)
        final_df = self.binarize_emotions(df)
        train_df, val_df, test_df = self.split_data(final_df)
        return train_df, val_df, test_df, self.mlb
