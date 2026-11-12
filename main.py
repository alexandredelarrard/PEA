from src.context import get_config_context
from src.data_aggregate.step_build_cube import StepBuildCube

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
# docker run --rm -v stock_pick_strat_pgdata:/volume -v "/c/Users/de larrard alexandre/Downloads:/backup:ro" alpine tar -xzf /backup/stock_pick_strat_pgdata.tar.gz -C /volume

# Save tlocally
#  docker run --rm -v stock_pick_strat_pgdata:/volume   -v "/D/:/backup" alpine tar -czf /backup/stock_pick_strat_pgdata.tar.gz -C /volume .
 