import paddle
__all__ = ['EMACallback']
import logging
from typing import Any, Dict
_logger = logging.getLogger(__name__)


>>>>>>class EMACallback(pytorch_lightning.callbacks.Callback):
    """
    Model Exponential Moving Average. Empirically it has been found that using the moving average
    of the trained parameters of a deep network is better than using its trained parameters directly.
    If `use_ema_weights`, then the ema parameters of the network is set after training end.
    """

    def __init__(self, decay=0.0001, use_ema_weights: bool=True):
        self.decay = 1.0 - decay
        self.ema = None
        self.use_ema_weights = use_ema_weights

    def on_fit_start(self, trainer, pl_module):
        """Initialize `ModelEmaV2` from timm to keep a copy of the moving average of the weights"""
>>>>>>        self.ema = timm.utils.model_ema.ModelEmaV2(pl_module, decay=self.
            decay, device=None)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx
        ):
        """Update the stored parameters using a moving average"""
        self.ema.update(pl_module)

    def on_validation_epoch_start(self, trainer, pl_module):
        """do validation using the stored parameters"""
        self.store(pl_module.parameters())
        self.copy_to(self.ema.module.parameters(), pl_module.parameters())

    def on_validation_end(self, trainer, pl_module):
        """Restore original parameters to resume training later"""
        self.restore(pl_module.parameters())

    def on_train_end(self, trainer, pl_module):
        if self.use_ema_weights:
            self.copy_to(self.ema.module.parameters(), pl_module.parameters())
            msg = 'Model weights replaced with the EMA version.'

    def state_dict(self):
        if self.ema is not None:
>>>>>>            return {'state_dict_ema': timm.utils.model.get_state_dict(self.
                ema, unwrap_model)}

    def load_state_dict(self, state_dict: Dict[str, Any]) ->None:
        if self.ema is not None:
            self.ema.module.set_state_dict(state_dict=state_dict[
                'state_dict_ema'])

    def store(self, parameters):
        """Save the current parameters for restoring later."""
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def copy_to(self, shadow_parameters, parameters):
        """Copy current parameters into given collection of parameters."""
        for s_param, param in zip(shadow_parameters, parameters):
            if not param.stop_gradient:
                param.data.copy_(s_param.data)
