from hydra.core.hydra_config import HydraConfig

def get_config():
    if HydraConfig.initialized():
        return HydraConfig.get().cfg
    raise RuntimeError("HydraConfig is not initialized. Call it within a Hydra application.")
