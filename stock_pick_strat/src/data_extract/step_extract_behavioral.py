"""
step_extract_behavioral.py  (src/data_extract/step_extract_behavioral.py)
-------------------------------------------------------------------------
Behavioral / retail-attention alt-data extraction:
  * Wikipedia pageviews (reliable & daily)
  * Google Trends (opt-in; needs `pip install pytrends`, self-skips if absent)
  * News (future)

Slow / rate-limited -> opt-in; the cube step picks up whatever parquet exists.
"""
from omegaconf import DictConfig

from src.context import Context
from src.utils.step import Step
from src.data_extract.utils.behavioral.fetch_wiki_pageviews import fetch_wiki_pageviews
from src.data_extract.utils.behavioral.fetch_google_trends import fetch_google_trends


class StepExtractBehavioral(Step):

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)

    def run(self, tickers: list[str]) -> None:
        fetch_wiki_pageviews(self._context, tickers=tickers)
        # fetch_google_trends(self._context, tickers=tickers)
