

from .lightningdit import LightningDiT_models
from .hidden_lightningdit import HiddenLightningDiT_models


gen_models = {
    **LightningDiT_models,
    **HiddenLightningDiT_models,
}