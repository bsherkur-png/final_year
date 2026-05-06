import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.ingestion.build_master_csv import build_master_csv


class TestDeduplicateUrls(unittest.TestCase):
    def test_build_master_csv_removes_duplicate_date_link_rows(self) -> None:
        input_rows = pd.DataFrame(
            [
                {
                    "source": "BBC",
                    "title": "First copy",
                    "link": "https://example.com/news/story-1",
                    "date": "2020-01-01",
                    "page": 1,
                    "snippet": "alpha",
                },
                {
                    "source": "The Guardian",
                    "title": "Duplicate copy",
                    "link": "https://example.com/news/story-1",
                    "date": "2020-01-02",
                    "page": 2,
                    "snippet": "beta",
                },
                {
                    "source": "Daily Mail",
                    "title": "Unique story",
                    "link": "https://example.com/news/story-2",
                    "date": "2020-01-03",
                    "page": 3,
                    "snippet": "gamma",
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_file = Path(tmp_dir) / "input.csv"
            output_file = Path(tmp_dir) / "output.csv"
            input_rows.to_csv(input_file, index=False)

            result = build_master_csv(input_file=input_file, output_file=output_file)

        self.assertEqual(len(result), 2)
        self.assertEqual(
            result["date_link"].tolist(),
            [
                "https://example.com/news/story-1",
                "https://example.com/news/story-2",
            ],
        )
        self.assertEqual(result["title"].tolist(), ["First copy", "Unique story"])
        self.assertEqual(result["article_id"].tolist(), [1, 2])


if __name__ == "__main__":
    unittest.main()
