from src.context import get_config_context
from src.modelling.steps.step_long_short import StepLongShort
from src.data_aggregate.step_build_cube import StepBuildCube
from src.data_extract.step_extract_all_data import StepExtractAllData


if __name__ == "__main__":
    config, context = get_config_context("./configs", use_cache=False, save=True)

    # self = StepExtractAllData(context=context, config=config)
    # self.run()

    self = StepBuildCube(context=context, config=config)
    # self.run(full=True)

    # self = StepLongShort(context=context, config=config)
    # self.run()

    # self = StepPortfolio(context=context, config=config)
    # self.run()

# load to docker
# docker run --rm -v stock_pick_strat_pgdata:/volume -vkz "/c/Users/alarr/Downloads:/backup:ro" alpine tar -xzf /backup/stock_pick_strat_pgdata.tar.gz -C /volume

# Save tlocally
#  docker run --rm -v stock_pick_strat_pgdata:/volume -v "/c/Users/alarr/Downloads:/backup" alpine tar -czf /backup/stock_pick_strat_pgdata.tar.gz -C /volume .
