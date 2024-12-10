import pandas as pd
from scipy.stats import pearsonr

class ConversationAnalyzer:
    def __init__(self, data):
        """
        Initialize the ConversationAnalyzer with the dataset.

        :param data: Path to the CSV file containing the conversation data.
        """
        self.data = data

    def compute_correlation(self, col1, col2):
        """
        Compute the correlation between two columns in the dataset.

        :param col1: The first column name.
        :param col2: The second column name.
        :return: Correlation coefficient and p-value.
        """
        if col1 not in self.data.columns or col2 not in self.data.columns:
            raise ValueError(f"Columns {col1} and/or {col2} not found in the dataset.")

        # Convert to numeric if necessary
        col1_data = pd.to_numeric(self.data[col1], errors='coerce')
        col2_data = pd.to_numeric(self.data[col2], errors='coerce')

        # Drop rows with NaN values (created by non-numeric conversion)
        valid_data = pd.DataFrame({col1: col1_data, col2: col2_data}).dropna()

        if valid_data.empty:
            raise ValueError("No valid data available for correlation computation.")

        # Compute Pearson correlation
        correlation, p_value = pearsonr(valid_data[col1], valid_data[col2])
        return correlation, p_value

    def analyze_position_distributions(self):
        """
        Analyze the likelihood of emotions, sentiment, and intensity appearing at different positions
        (beginning, middle, end) of the conversation.

        :return: A dictionary with positional likelihoods for sentiment, emotion, and intensity.
        """
        total_rows = len(self.data)
        thresholds = {
            'beginning': (0, total_rows // 3),
            'middle': (total_rows // 3, 2 * total_rows // 3),
            'end': (2 * total_rows // 3, total_rows)
        }

        results = {}
        for position, (start, end) in thresholds.items():
            subset = self.data.iloc[start:end]
            results[position] = {
                'sentiment': subset['sentiment'].value_counts(normalize=True).to_dict(),
                'emotion': subset['emotion'].value_counts(normalize=True).to_dict(),
                'intensity': subset['intensity'].value_counts(normalize=True).to_dict()
            }

        return results

    def predict_next_state(self, current_emotion, current_intensity):
        """
        Predict the likely next emotion and intensity based on the current state.

        :param current_emotion: Current emotion.
        :param current_intensity: Current intensity.
        :return: Likely next emotion and intensity.
        """
        transitions = self.data[['emotion', 'intensity']].shift(-1)
        current_states = self.data[['emotion', 'intensity']]

        mask = (current_states['emotion'] == current_emotion) & (current_states['intensity'] == current_intensity)
        next_states = transitions[mask]

        if not next_states.empty:
            next_emotion = next_states['emotion'].mode().iloc[0] if not next_states['emotion'].mode().empty else None
            next_intensity = next_states['intensity'].mode().iloc[0] if not next_states['intensity'].mode().empty else None
            return next_emotion, next_intensity
        else:
            return None, None

    def analyze_emotion_transitions(self):
        """
        Analyze transitions between emotions, sentiment, and intensity over the conversation.

        :return: A DataFrame summarizing transitions.
        """
        self.data['next_emotion'] = self.data['emotion'].shift(-1)
        self.data['next_intensity'] = self.data['intensity'].shift(-1)

        transition_summary = self.data.groupby(['emotion', 'intensity', 'next_emotion', 'next_intensity']).size()
        return transition_summary.reset_index(name='count')

# Example Usage
if __name__ == "__main__":
    # Initialize the analyzer
    analyzer = ConversationAnalyzer("D:/julixus/data/meisd_project/data/MEISD_text.csv")

    analyzer.data['sentiment'] = analyzer.data['sentiment'].map({
        "positive": 1,
        "negative": -1,
        "neutral": 0
    })

    # Ensure intensity is numeric
    analyzer.data['intensity'] = pd.to_numeric(analyzer.data['intensity'], errors='coerce')
    # Compute correlation between sentiment and intensity
    correlation, p_value = analyzer.compute_correlation("sentiment", "intensity")
    print(f"Correlation between sentiment and intensity: {correlation} (p-value: {p_value})")

    # Analyze position distributions
    position_analysis = analyzer.analyze_position_distributions()
    print("Position Analysis:", position_analysis)

    # Predict the next state given an emotion and intensity
    next_emotion, next_intensity = analyzer.predict_next_state("anger", 5)
    print(f"Likely next emotion: {next_emotion}, Likely next intensity: {next_intensity}")

    # Analyze emotion transitions
    transitions = analyzer.analyze_emotion_transitions()
    print("Emotion Transitions:")
    print(transitions)
