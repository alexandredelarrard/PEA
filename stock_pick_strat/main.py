from src.context import get_config_context
from src.data_extract.step_extract_all_data import StepExtractAllData
from src.data_aggregate.step_build_cube import StepBuildCube
from src.modelling.step_modelling import StepModelling
from src.post_processing.step_backtest import StepBacktest

if __name__ == "__main__":
    config, context = get_config_context("./configs", use_cache=False, save=True)
    
    # self = StepExtractAllData(context=context, config=config)
    # self.run()

    # self = StepBuildCube(context=context, config=config)
    # self.run()

    # self = StepModelling(context=context, config=config)
    # self.run()

    self = StepBacktest(context=context, config=config)
    self.run() 