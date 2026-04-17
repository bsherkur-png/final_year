import pandas as pd
from src.pipeline.news_pipeline import NewsPipeline

pipeline = NewsPipeline()

# Before filter
master_df = pd.read_csv(pipeline.ingestion_output_path)

# After extraction (this applies ShamimaBegumFilter in your current code)
filtered_df = pipeline.run_extraction()

removed_df = master_df.loc[
    ~master_df["article_id"].isin(filtered_df["article_id"])
].copy()

print(f"master rows:   {len(master_df)}")
print(f"kept rows:     {len(filtered_df)}")
print(f"removed rows:  {len(removed_df)}")

print(removed_df[["article_id", "news_outlet", "title", "date_link"]].head(20))
