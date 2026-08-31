from src.context import get_config_context
 
from src.data_extract.step_extract_all_data import StepExtractAllData
from src.data_aggregate.step_build_cube import StepBuildCube
from src.modelling.long_short.step_train import StepModelling
from src.strategies.step_super_investors import SuperInvestorsStrategy
from src.portfolio import StepPortfolio

if __name__ == "__main__":
    config, context = get_config_context("./configs", use_cache=False, save=True)

    self = StepExtractAllData(context=context, config=config)
    # self.run()

    # self = StepBuildCube(context=context, config=config)
    # self.run(full=True)

    # self = SuperInvestorsStrategy(context=context, config=config).run()

    # self = StepModelling(context=context, config=config)
    # self.run()

    # self = StepPortfolio(context=context, config=config)
    # self.run()

##### 

# extraction date - enrich
# TODO: short interest : extract also the Lit exchange NYSE /NASDAQ, from 2009 for all (now is 2018)  
# TODO: insiders trading : sec form 3/4/5 -> sec since 2003, zip since Q1 2006 -> take sec 
# TODO: earnings surprises starts 1999-08, but empty till ~2003

# other 
# TODO: check data is consistent over time, even for latest 2026 month ? 
# TODO: what happens when new data pops between quarters on financials 
# TODO: include move from peers when new results are avilable -> move all peers info
# TODO: refine modelling to be as stable as possible
# TODO: review periods when IC drops for few weeks / months
# TODO: check if all neutrality is correctly done on target 
# TODO: add other strats decorrelated : - Super investors replica ? 
# TODO: put horizons to 21,42,63 since we a  re in open days, not calendar days

# docker run --rm -v database_pgdata:/volume alpine tar czf - -C /volume . > D:/database_pgdata.tar.gz

##### timeline 

########### 27/07/26
# [2026-07-27, 00:10:39 UTC] {subprocess.py:106} INFO - 2026-07-27 00:10:39 - src.utils.step - INFO - step_train.py - horizon 90: [ENSEMBLE] CV mean_IC=+0.0443  IC_IR=+1.45
# [2026-07-27, 00:10:39 UTC] {subprocess.py:106} INFO - 2026-07-27 00:10:39 - src.utils.step - INFO - step_train.py - horizon 90:   [elasticnet] CV mean_IC=+0.0424  IC_IR=+1.28
# [2026-07-27, 00:10:39 UTC] {subprocess.py:106} INFO - 2026-07-27 00:10:39 - src.utils.step - INFO - step_train.py - horizon 90:   [lgbm      ] CV mean_IC=+0.0281  IC_IR=+1.06
# [2026-07-27, 00:10:39 UTC] {subprocess.py:106} INFO - 2026-07-27 00:10:39 - src.utils.step - INFO - step_train.py - horizon 90:   [random_forest] CV mean_IC=+0.0258  IC_IR=+0.87

# [2026-07-27, 00:04:24 UTC] {subprocess.py:106} INFO - 2026-07-27 00:04:24 - src.utils.step - INFO - step_train.py - horizon 60: [ENSEMBLE] CV mean_IC=+0.0453  IC_IR=+1.87
# [2026-07-27, 00:04:24 UTC] {subprocess.py:106} INFO - 2026-07-27 00:04:24 - src.utils.step - INFO - step_train.py - horizon 60:   [elasticnet] CV mean_IC=+0.0424  IC_IR=+1.50
# [2026-07-27, 00:04:24 UTC] {subprocess.py:106} INFO - 2026-07-27 00:04:24 - src.utils.step - INFO - step_train.py - horizon 60:   [lgbm      ] CV mean_IC=+0.0270  IC_IR=+1.18
# [2026-07-27, 00:04:24 UTC] {subprocess.py:106} INFO - 2026-07-27 00:04:24 - src.utils.step - INFO - step_train.py - horizon 60:   [random_forest] CV mean_IC=+0.0334  IC_IR=+1.29

# [2026-07-26, 23:58:20 UTC] {subprocess.py:106} INFO - 2026-07-26 23:58:20 - src.utils.step - INFO - step_train.py - horizon 30: [ENSEMBLE] CV mean_IC=+0.0404  IC_IR=+2.04
# [2026-07-26, 23:58:20 UTC] {subprocess.py:106} INFO - 2026-07-26 23:58:20 - src.utils.step - INFO - step_train.py - horizon 30:   [elasticnet] CV mean_IC=+0.0325  IC_IR=+1.59
# [2026-07-26, 23:58:20 UTC] {subprocess.py:106} INFO - 2026-07-26 23:58:20 - src.utils.step - INFO - step_train.py - horizon 30:   [lgbm      ] CV mean_IC=+0.0272  IC_IR=+1.42
# [2026-07-26, 23:58:20 UTC] {subprocess.py:106} INFO - 2026-07-26 23:58:20 - src.utils.step - INFO - step_train.py - horizon 30:   [random_forest] CV mean_IC=+0.0333  IC_IR=+1.94