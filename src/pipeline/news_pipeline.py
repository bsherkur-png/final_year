from pathlib import Path

import pandas as pd

from src.extraction.scraper import Extractor


class Pipeline:
    def __init__(self, source, extractor=None):
        self.df = None
        self.source_path = Path(source)
        self.extractor = extractor or Extractor()

    def run(self):
        self.df = pd.read_csv(self.source_path)
        self.df = self.extractor.extract(self.df)
        return self.df


def run_news_pipeline(source):
    pipeline = Pipeline(source)
    return pipeline.run()
