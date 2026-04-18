import pandas as pd


class OutletComparator:
    def summarize_outlets(
        self, df: pd.DataFrame, polarity_column: str = "vader_score"
    ) -> pd.DataFrame:
        if "news_outlet" not in df.columns:
            raise ValueError('Missing required column: "news_outlet"')
        if polarity_column not in df.columns:
            raise ValueError(f'Missing required column: "{polarity_column}"')

        score_columns = ("vader_score", "sentiwordnet_score", "nrc_score")
        present_columns = [column for column in score_columns if column in df.columns]

        if not present_columns:
            return df[["news_outlet"]].drop_duplicates().reset_index(drop=True)

        aggregations = {column: ["mean", "std", "count"] for column in present_columns}
        summary_df = df.groupby("news_outlet", dropna=False).agg(aggregations).reset_index()

        summary_df.columns = [
            "_".join(str(part) for part in parts if str(part))
            if isinstance(parts, tuple)
            else str(parts)
            for parts in summary_df.columns
        ]
        return summary_df
