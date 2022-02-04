import dotenv
import hydra
from omegaconf import DictConfig
from rich.traceback import install

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
from tali.run_data_only import sample_datamodule
from tali.sample_actuate_data import sample_and_upload_datamodule

dotenv.load_dotenv(override=True)
install(show_locals=True, extra_lines=1, word_wrap=True, width=350)


@hydra.main(config_path="configs", config_name="config")
def main(config: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from base import utils
    from tali.train import train_eval

    # A couple of optional utilities:
    # - disabling python warnings
    # - forcing debug-friendly configuration
    # - verifying experiment name is set when running in experiment mode
    # You can safely get rid of this line if you don't want those
    utils.extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    if config.debug_data:
        # iterate through dataloaders only with current config
        # -- used to test datamodules
        return sample_datamodule(config)
    elif config.wandb_visualization_config.visualize_data_in_wandb:
        return sample_and_upload_datamodule(config)
    else:
        # Train model in a single run
        return train_eval(config)


if __name__ == "__main__":
    main()
