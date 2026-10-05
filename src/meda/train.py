"""Hydra entry point for training (SPEC section 13).

Usage: `python -m meda.train agent=ppo generator=tiny_dilution`. Training loops arrive in M6/M8.
"""

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    raise NotImplementedError("Training is implemented in M6 (tabular Q) and M8 (PPO).")


if __name__ == "__main__":
    main()
