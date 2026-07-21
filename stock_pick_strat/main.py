from src.context import get_config_context
from src.data_extract.step_extract_all_data import StepExtractAllData
from src.data_aggregate.step_build_cube import StepBuildCube
from src.modelling.step_modelling import StepModelling
from src.post_processing.step_backtest import StepBacktest
from src.data_peers.step_deduce_peers import StepDeducePeers

if __name__ == "__main__":
    config, context = get_config_context("./configs", use_cache=False, save=True)
    
    # self = StepExtractAllData(context=context, config=config)
    # self.run()

    # self = StepDeducePeers(context=context, config=config)
    # self.run()

    # self = StepBuildCube(context=context, config=config)
    # self.run()

    self = StepModelling(context=context, config=config)
    self.run()

    # self = StepBacktest(context=context, config=config)
    # self.run() 

# TODO: check how to rebase google trend week after week if pick is now ? 
# TODO, clean def 14 to deduce fields missing : for instance if large total comp but missing info on bonus, but similar past year, then cen fill missing value, etc.
# TODO: check why those features are not computed 
# TODO: check Configured features not in cube (skipped): ['f_gross_profitability_xs', 'f_interest_coverage_vs_peers', 'f_net_debt_to_ebitda_xs', 'f_ec_tone_xs', 'f_ec_tone_delta_xs', 'f_ec_qa_gap_xs', 'f_ec_uncertainty_xs', 'f_ec_vocab_novelty_xs', 'f_ec_length_delta_xs']

# TODO ##################### biggeer movers :
# - Add News and deduce : geopolitics score per stock
# - Neutral currency pools: 
#           - LLM text extract from form 8 to get geo weight to build currency basket
#           - LLM to build commo pool impact  
# - Earnings call transcript analysis 
# - Add notes text & nums analysis to the cube 
# - Impact of correlated, close peers financials arriving before with earnings -> peers compute will move with their earnings 

# docker run --rm -v stock_pick_strat_pgdata:/volume alpine tar czf - -C /volume . > stock_pick_strat_pgdata.tar.gz