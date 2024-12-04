import os
import paddle
from typing import List, Optional, Tuple
from uuid import uuid4
import shutil
from pl_trainer import DDPMModule
from oa_reactdiff.trainer.ema import EMACallback
from oa_reactdiff.model import EGNN, LEFTNet
model_type = 'leftnet'
version = '0'
project = 'OAReactDiff'
egnn_config = dict(in_node_nf=8, in_edge_nf=0, hidden_nf=256,
    edge_hidden_nf=64, act_fn='swish', n_layers=9, attention=True,
    out_node_nf=None, tanh=True, coords_range=15.0, norm_constant=1.0,
    inv_sublayers=1, sin_embedding=True, normalization_factor=1.0,
    aggregation_method='mean')
leftnet_config = dict(pos_require_grad=False, cutoff=10.0, num_layers=6,
    hidden_channels=196, num_radial=96, in_hidden_channels=8, reflect_equiv
    =True, legacy=True, update=True, pos_grad=False, single_layer_output=
    True, object_aware=True)
if model_type == 'leftnet':
    model_config = leftnet_config
    model = LEFTNet
elif model_type == 'egnn':
    model_config = egnn_config
    model = EGNN
else:
    raise KeyError('model type not implemented.')
optimizer_config = dict(lr=0.00025, betas=[0.9, 0.999], weight_decay=0,
    amsgrad=True)
T_0 = 200
T_mult = 2
training_config = dict(datadir='../data/transition1x/', remove_h=False, bz=
    14, num_workers=0, clip_grad=True, gradient_clip_val=None, ema=False,
    ema_decay=0.999, swapping_react_prod=True, append_frag=False,
    use_by_ind=True, reflection=False, single_frag_only=True, only_ts=False,
    lr_schedule_type=None, lr_schedule_config=dict(gamma=0.8, step_size=100))
training_data_frac = 1.0
node_nfs: List[int] = [9] * 3
edge_nf: int = 0
condition_nf: int = 1
fragment_names: List[str] = ['R', 'TS', 'P']
pos_dim: int = 3
update_pocket_coords: bool = True
condition_time: bool = True
edge_cutoff: Optional[float] = None
loss_type = 'l2'
pos_only = True
process_type = 'TS1x'
enforce_same_encoding = None
scales = [1.0, 2.0, 1.0]
fixed_idx: Optional[List] = None
eval_epochs = 10
norm_values: Tuple = (1.0, 1.0, 1.0)
norm_biases: Tuple = (0.0, 0.0, 0.0)
noise_schedule: str = 'cosine'
timesteps: int = 5000
precision: float = 1e-05
norms = '_'.join([str(x) for x in norm_values])
run_name = f'{model_type}-{version}-' + str(uuid4()).split('-')[-1]
>>>>>>pytorch_lightning.seed_everything(42, workers=True)
ddpm = DDPMModule(model_config, optimizer_config, training_config, node_nfs,
    edge_nf, condition_nf, fragment_names, pos_dim, update_pocket_coords,
    condition_time, edge_cutoff, norm_values, norm_biases, noise_schedule,
    timesteps, precision, loss_type, pos_only, process_type, model,
    enforce_same_encoding, scales, source=None, fixed_idx=fixed_idx,
    eval_epochs=eval_epochs)
config = model_config.copy()
config.update(optimizer_config)
config.update(training_config)
trainer = None
>>>>>>if trainer is None or isinstance(trainer, pytorch_lightning.Trainer
    ) and trainer.is_global_zero:
>>>>>>    wandb_logger = pytorch_lightning.loggers.WandbLogger(project=project,
        log_model=False, name=run_name)
    try:
        wandb_logger.experiment.config.update(config)
        wandb_logger.watch(ddpm.ddpm.dynamics, log='all', log_freq=100,
            log_graph=False)
    except:
        pass
ckpt_path = f'checkpoint/{project}/{wandb_logger.experiment.name}'
>>>>>>earlystopping = pytorch_lightning.callbacks.EarlyStopping(monitor=
    'val-totloss', patience=2000, verbose=True, log_rank_zero_only=True)
>>>>>>checkpoint_callback = pytorch_lightning.callbacks.ModelCheckpoint(monitor=
    'val-totloss', dirpath=ckpt_path, filename=
    'ddpm-{epoch:03d}-{val-totloss:.2f}', every_n_epochs=1, save_top_k=-1)
>>>>>>lr_monitor = pytorch_lightning.callbacks.LearningRateMonitor(logging_interval
    ='step')
>>>>>>callbacks = [earlystopping, checkpoint_callback, pytorch_lightning.
    callbacks.progress.TQDMProgressBar(), lr_monitor]
if training_config['ema']:
    callbacks.append(EMACallback(decay=training_config['ema_decay']))
if not os.path.isdir(ckpt_path):
    os.makedirs(ckpt_path)
shutil.copy(f'../model/{model_type}.py', f'{ckpt_path}/{model_type}.py')
print('config: ', config)
strategy = None
devices = [0]
>>>>>>strategy = pytorch_lightning.strategies.ddp.DDPStrategy(find_unused_parameters
    =True)
if strategy is not None:
    devices = list(range(paddle.device.cuda.device_count()))
if len(devices) == 1:
    strategy = None
>>>>>>trainer = pytorch_lightning.Trainer(max_epochs=2000, accelerator='gpu',
    deterministic=False, devices=devices, strategy=strategy,
    log_every_n_steps=1, callbacks=callbacks, profiler=None, logger=
    wandb_logger, accumulate_grad_batches=1, gradient_clip_val=
    training_config['gradient_clip_val'], limit_train_batches=200,
    limit_val_batches=20)
trainer.fit(ddpm)
trainer.save_checkpoint('pretrained-ts1x-diff.ckpt')
