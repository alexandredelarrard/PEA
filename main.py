from src.context import get_config_context
from src.data_aggregate.step_build_cube import StepBuildCube

if __name__ == "__main__":
    config, context = get_config_context("./configs", use_cache=False, save=True)

    # self = StepExtractAllData(context=context, config=config)
    # self.run()

    self = StepBuildCube(context=context, config=config)
    # self.run(full=True)

    # self = StepModelling(context=context, config=config)
    # self.run()

    # self = StepPortfolio(context=context, config=config)
    # self.run()

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
