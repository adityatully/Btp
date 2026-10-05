"""Hydra entry point for evaluation (SPEC section 12.2).

Usage: `python -m meda.evaluate_cli producer=<name> benchmark=<set>`. Implemented in M7.
"""

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    raise NotImplementedError("Evaluation is implemented in M7.")


if __name__ == "__main__":
    main()
