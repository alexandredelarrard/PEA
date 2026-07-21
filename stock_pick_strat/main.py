# Trust the corporate TLS-inspection proxy CA. MUST run before importing any HTTP
# client: yfinance/curl_cffi freeze their default CA path at import time, so setting
# the CA env vars afterwards would be ignored. No-op if you already set SSL_CERT_FILE
# yourself (see src/utils/ssl_setup.py; `python -m src.utils.ssl_setup` for setx).
from src.utils.ssl_setup import configure_corporate_ca
configure_corporate_ca()

from src.context import get_config_context
from src.data_extract.step_extract_all_data import StepExtractAllData
from src.data_aggregate.step_build_cube import StepBuildCube
from src.modelling.step_modelling import StepModelling
from src.post_processing.step_backtest import StepBacktest
from src.data_peers.step_deduce_peers import StepDeducePeers

if __name__ == "__main__":
    config, context = get_config_context("./configs", use_cache=False, save=True)
    
    self = StepExtractAllData(context=context, config=config)
    self.run()

    # self = StepDeducePeers(context=context, config=config)
    # self.run()

    # self = StepBuildCube(context=context, config=config)
    # self.run()

    # self = StepModelling(context=context, config=config)
    # self.run()

    # self = StepBacktest(context=context, config=config)
    # self.run() 

# TODO: look at macro for today day, maybe 1 day lag always...
# TODO: check 498 tickers in employees, missing 'FDXF', 'HONA'
# 125 tickers have pension facts -> industry mostly
# TODO: insider gives shares / stock of CEO, COO, CFO etc acquired and price. stock op plan. TO derive place
# self.short_interest only starts in 2018 ?
# 499 failed to deliver at least once, huge pick in 2026 ? 
# TODO: why only 473 tickers for wiki ? 
# 'MTB', 'FOXA', 'TTD', 'SJM', 'DE', 'COO', 'SPGI', 'TRV', 'MKC', 'ORLY', 'LLY', 'PCG', 'MOS', 'PEG', 'TAP', 'PKG', 'WEC', 'HD', 'EL', 'KO', 'GOOGL', 'KKR', 'AJG', 'NWSA',  'HIG',  'DIS', 'HSY'},
# TODO: check how to rebase google trend week after week if pick is now ? 
# TODO, clean def 14 to deduce fields missing : for instance if large total comp but missing info on bonus, but similar past year, then cen fill missing value, etc.
# 413 with divid only ? -> might be because no div for some of them -> should still be there, at 0
# TODO: outstanding div, sometimes happend. How to handle the increase std and mean over time ? 

# docker run --rm -v stock_pick_strat_pgdata:/volume alpine tar czf - -C /volume . > stock_pick_strat_pgdata.tar.gz