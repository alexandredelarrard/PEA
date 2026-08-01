from src.context import get_config_context
from src.data_extract.step_extract_all_data import StepExtractAllData
# from src.data_aggregate.step_build_cube import StepBuildCube
# from src.modelling.long_short.step_train import StepModelling
# from src.portfolio import StepPortfolio
# from src.data_peers.step_deduce_peers import StepDeducePeers

# from src.data_extract.utils.fundamentals.fetch_fundamentals_edgar import fetch_fundamentals_edgartools

if __name__ == "__main__":
    config, context = get_config_context("./configs", use_cache=False, save=True)

    # fetch_fundamentals_edgartools(context, tickers=['MAA', 'A', 'AEE', 'ODFL', 'JPM'])

    self = StepExtractAllData(context=context, config=config)
    self.run()

    # self = StepDeducePeers(context=context, config=config)
    # self.run()ss

    # self = StepBuildCube(context=context, config=config)
    # self.run()

    # self = StepModelling(context=context, config=config)
    # self.run()

    # self = StepPortfolio(context=context, config=config)
    # self.run()

##### TODO
# TODO: check data is consistent over time, even for latest 2026 month ? 
# TODO: what happens when new data pops between quarters on financials 
# TODO: include move from peers when new results are avilable -> move all peers info
# TODO: refine modelling to be as stable as possible
# TODO: review periods when IC drops for few weeks / months
# TODO: check if all neutrality is correctly done on target 
# TODO: add other strats decorrelated : - Super investors replica ? 
# TODO: put horizons to 21,42,63 since we are in open days, not calendar days

# docker run --rm -v database_pgdata:/volume alpine tar czf - -C /volume . > D:/database_pgdata.tar.gz

##### timeline 
# sec insider = + 45days after as_of +1 16august for 30 june 

# 2026-07-22 22:40:03 - src.utils.step - INFO - step_modelling.py - horizon 90: [ENSEMBLE] CV mean_IC=+0.0463  IC_IR=+1.50
# 2026-07-22 22:40:03 - src.utils.step - INFO - step_modelling.py - horizon 90:   [elasticnet] CV mean_IC=+0.0314  IC_IR=+1.12
# 2026-07-22 22:40:03 - src.utils.step - INFO - step_modelling.py - horizon 90:   [lgbm      ] CV mean_IC=+0.0342  IC_IR=+1.13
# 2026-07-22 22:40:03 - src.utils.step - INFO - step_modelling.py - horizon 90:   [random_forest] CV mean_IC=+0.0329  IC_IR=+1.28

# 2026-07-22 22:37:04 - src.utils.step - INFO - step_modelling.py - horizon 60: [ENSEMBLE] CV mean_IC=+0.0427  IC_IR=+1.55
# 2026-07-22 22:37:04 - src.utils.step - INFO - step_modelling.py - horizon 60:   [elasticnet] CV mean_IC=+0.0349  IC_IR=+1.32
# 2026-07-22 22:37:04 - src.utils.step - INFO - step_modelling.py - horizon 60:   [lgbm      ] CV mean_IC=+0.0250  IC_IR=+1.02
# 2026-07-22 22:37:04 - src.utils.step - INFO - step_modelling.py - horizon 60:   [random_forest] CV mean_IC=+0.0347  IC_IR=+1.42

# 2026-07-22 22:34:40 - src.utils.step - INFO - step_modelling.py - horizon 30: [ENSEMBLE] CV mean_IC=+0.0403  IC_IR=+2.07
# 2026-07-22 22:34:40 - src.utils.step - INFO - step_modelling.py - horizon 30:   [elasticnet] CV mean_IC=+0.0312  IC_IR=+1.68
# 2026-07-22 22:34:40 - src.utils.step - INFO - step_modelling.py - horizon 30:   [lgbm      ] CV mean_IC=+0.0265  IC_IR=+1.32
# 2026-07-22 22:34:40 - src.utils.step - INFO - step_modelling.py - horizon 30:   [random_forest] CV mean_IC=+0.0343  IC_IR=+1.90


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

# docker exec -it pea_db vacuumdb -U alexandre -d pea --full
