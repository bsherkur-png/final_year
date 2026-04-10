import pandas as pd


class TopicVisualizer:
    def build_topic_count_frame(self, df: pd.DataFrame, topic_column: str = "topic_label") -> pd.DataFrame:
        if topic_column not in df.columns:
            raise ValueError(f"Missing required column: {topic_column}")

        return (
            df[topic_column]
            .fillna("Unlabelled")
            .value_counts()
            .rename_axis(topic_column)
            .reset_index(name="count")
        )

    def plot_topic_counts(self, counts_df: pd.DataFrame, topic_column: str = "topic_label"):
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots()
        axis.bar(counts_df[topic_column], counts_df["count"])
        axis.set_ylabel("Count")
        axis.set_xlabel("Topic")
        axis.set_title("Articles per Topic")
        axis.tick_params(axis="x", rotation=45)
        figure.tight_layout()
        return figure
