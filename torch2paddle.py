import numpy as np
import torch
import paddle

def torch2paddle():
    # torch_path = "csp_torch.pth"
    torch_path = "./pretrained-ts1x-diff.ckpt"
    paddle_path = "./pretrained-ts1x-diff.pdparams"
    torch_state_dict = torch.load(torch_path)
    fc_names = [
        'ddpm.dynamics.model.embedding',
        'ddpm.dynamics.model.embedding_out',
        'ddpm.dynamics.model.neighbor_emb.embedding',
        'ddpm.dynamics.model.s2v.lin1.0',
        'ddpm.dynamics.model.radial_lin.0',
        'ddpm.dynamics.model.radial_lin.2',
        'ddpm.dynamics.model.lin3.0',
        'ddpm.dynamics.model.lin3.2',
        'ddpm.dynamics.model.pos_expansion.mlp.0.linear',
        'ddpm.dynamics.model.pos_expansion.mlp.1.linear',
        'ddpm.dynamics.model.distance_embedding.mlp.0.linear',
        'ddpm.dynamics.model.distance_embedding.mlp.1.linear',
        'ddpm.dynamics.model.gcl_layers.0.edge_mlp.mlp.0.linear',
        'ddpm.dynamics.model.gcl_layers.0.edge_mlp.mlp.1.linear',
        'ddpm.dynamics.model.gcl_layers.0.node_mlp.mlp.0.linear',
        'ddpm.dynamics.model.gcl_layers.0.node_mlp.mlp.1.linear',
        'ddpm.dynamics.model.gcl_layers.0.edge_out_trans.mlp.0.linear',
        'ddpm.dynamics.model.gcl_layers.0.att_mlp.mlp.0.linear',
        'ddpm.dynamics.model.gcl_layers.1.edge_mlp.mlp.0.linear',
        'ddpm.dynamics.model.gcl_layers.1.edge_mlp.mlp.1.linear',
        'ddpm.dynamics.model.gcl_layers.1.node_mlp.mlp.0.linear',
        'ddpm.dynamics.model.gcl_layers.1.node_mlp.mlp.1.linear',
        'ddpm.dynamics.model.gcl_layers.1.edge_out_trans.mlp.0.linear',
        'ddpm.dynamics.model.gcl_layers.1.att_mlp.mlp.0.linear',
        'ddpm.dynamics.model.gcl_layers.2.edge_mlp.mlp.0.linear',
        'ddpm.dynamics.model.gcl_layers.2.edge_mlp.mlp.1.linear',
        'ddpm.dynamics.model.gcl_layers.2.node_mlp.mlp.0.linear',
        'ddpm.dynamics.model.gcl_layers.2.node_mlp.mlp.1.linear',
        'ddpm.dynamics.model.gcl_layers.2.edge_out_trans.mlp.0.linear',
        'ddpm.dynamics.model.gcl_layers.2.att_mlp.mlp.0.linear',
        'ddpm.dynamics.model.gcl_layers.3.edge_mlp.mlp.0.linear',
        'ddpm.dynamics.model.gcl_layers.3.edge_mlp.mlp.1.linear',
        'ddpm.dynamics.model.gcl_layers.3.node_mlp.mlp.0.linear',
        'ddpm.dynamics.model.gcl_layers.3.node_mlp.mlp.1.linear',
        'ddpm.dynamics.model.gcl_layers.3.edge_out_trans.mlp.0.linear',
        'ddpm.dynamics.model.gcl_layers.3.att_mlp.mlp.0.linear',
        'ddpm.dynamics.model.gcl_layers.4.edge_mlp.mlp.0.linear',
        'ddpm.dynamics.model.gcl_layers.4.edge_mlp.mlp.1.linear',
        'ddpm.dynamics.model.gcl_layers.4.node_mlp.mlp.0.linear',
        'ddpm.dynamics.model.gcl_layers.4.node_mlp.mlp.1.linear',
        'ddpm.dynamics.model.gcl_layers.4.edge_out_trans.mlp.0.linear',
        'ddpm.dynamics.model.gcl_layers.4.att_mlp.mlp.0.linear',
        'ddpm.dynamics.model.gcl_layers.5.edge_mlp.mlp.0.linear',
        'ddpm.dynamics.model.gcl_layers.5.edge_mlp.mlp.1.linear',
        'ddpm.dynamics.model.gcl_layers.5.node_mlp.mlp.0.linear',
        'ddpm.dynamics.model.gcl_layers.5.node_mlp.mlp.1.linear',
        'ddpm.dynamics.model.gcl_layers.5.edge_out_trans.mlp.0.linear',
        'ddpm.dynamics.model.gcl_layers.5.att_mlp.mlp.0.linear',
        'ddpm.dynamics.model.message_layers.0.dir_proj.0',
        'ddpm.dynamics.model.message_layers.0.dir_proj.2',
        'ddpm.dynamics.model.message_layers.0.x_proj.0',
        'ddpm.dynamics.model.message_layers.0.x_proj.2',
        'ddpm.dynamics.model.message_layers.0.rbf_proj',
        'ddpm.dynamics.model.message_layers.1.dir_proj.0',
        'ddpm.dynamics.model.message_layers.1.dir_proj.2',
        'ddpm.dynamics.model.message_layers.1.x_proj.0',
        'ddpm.dynamics.model.message_layers.1.x_proj.2',
        'ddpm.dynamics.model.message_layers.1.rbf_proj',
        'ddpm.dynamics.model.message_layers.2.dir_proj.0',
        'ddpm.dynamics.model.message_layers.2.dir_proj.2',
        'ddpm.dynamics.model.message_layers.2.x_proj.0',
        'ddpm.dynamics.model.message_layers.2.x_proj.2',
        'ddpm.dynamics.model.message_layers.2.rbf_proj',
        'ddpm.dynamics.model.message_layers.3.dir_proj.0',
        'ddpm.dynamics.model.message_layers.3.dir_proj.2',
        'ddpm.dynamics.model.message_layers.3.x_proj.0',
        'ddpm.dynamics.model.message_layers.3.x_proj.2',
        'ddpm.dynamics.model.message_layers.3.rbf_proj',
        'ddpm.dynamics.model.message_layers.4.dir_proj.0',
        'ddpm.dynamics.model.message_layers.4.dir_proj.2',
        'ddpm.dynamics.model.message_layers.4.x_proj.0',
        'ddpm.dynamics.model.message_layers.4.x_proj.2',
        'ddpm.dynamics.model.message_layers.4.rbf_proj',
        'ddpm.dynamics.model.message_layers.5.dir_proj.0',
        'ddpm.dynamics.model.message_layers.5.dir_proj.2',
        'ddpm.dynamics.model.message_layers.5.x_proj.0',
        'ddpm.dynamics.model.message_layers.5.x_proj.2',
        'ddpm.dynamics.model.message_layers.5.rbf_proj',
        'ddpm.dynamics.model.update_layers.0.vec_proj',
        'ddpm.dynamics.model.update_layers.0.xvec_proj.0',
        'ddpm.dynamics.model.update_layers.0.xvec_proj.2',
        'ddpm.dynamics.model.update_layers.0.lin3.0',
        'ddpm.dynamics.model.update_layers.0.lin3.2',
        'ddpm.dynamics.model.update_layers.0.lin3.4',
        'ddpm.dynamics.model.update_layers.1.vec_proj',
        'ddpm.dynamics.model.update_layers.1.xvec_proj.0',
        'ddpm.dynamics.model.update_layers.1.xvec_proj.2',
        'ddpm.dynamics.model.update_layers.1.lin3.0',
        'ddpm.dynamics.model.update_layers.1.lin3.2',
        'ddpm.dynamics.model.update_layers.1.lin3.4',
        'ddpm.dynamics.model.update_layers.2.vec_proj',
        'ddpm.dynamics.model.update_layers.2.xvec_proj.0',
        'ddpm.dynamics.model.update_layers.2.xvec_proj.2',
        'ddpm.dynamics.model.update_layers.2.lin3.0',
        'ddpm.dynamics.model.update_layers.2.lin3.2',
        'ddpm.dynamics.model.update_layers.2.lin3.4',
        'ddpm.dynamics.model.update_layers.3.vec_proj',
        'ddpm.dynamics.model.update_layers.3.xvec_proj.0',
        'ddpm.dynamics.model.update_layers.3.xvec_proj.2',
        'ddpm.dynamics.model.update_layers.3.lin3.0',
        'ddpm.dynamics.model.update_layers.3.lin3.2',
        'ddpm.dynamics.model.update_layers.3.lin3.4',
        'ddpm.dynamics.model.update_layers.4.vec_proj',
        'ddpm.dynamics.model.update_layers.4.xvec_proj.0',
        'ddpm.dynamics.model.update_layers.4.xvec_proj.2',
        'ddpm.dynamics.model.update_layers.4.lin3.0',
        'ddpm.dynamics.model.update_layers.4.lin3.2',
        'ddpm.dynamics.model.update_layers.4.lin3.4',
        'ddpm.dynamics.model.update_layers.5.vec_proj',
        'ddpm.dynamics.model.update_layers.5.xvec_proj.0',
        'ddpm.dynamics.model.update_layers.5.xvec_proj.2',
        'ddpm.dynamics.model.update_layers.5.lin3.0',
        'ddpm.dynamics.model.update_layers.5.lin3.2',
        'ddpm.dynamics.model.update_layers.5.lin3.4',
        'ddpm.dynamics.model.last_layer',
        'ddpm.dynamics.model.out_pos.output_network.0.vec1_proj',
        'ddpm.dynamics.model.out_pos.output_network.0.vec2_proj',
        'ddpm.dynamics.model.out_pos.output_network.0.update_net.0',
        'ddpm.dynamics.model.out_pos.output_network.0.update_net.2',
        'ddpm.dynamics.encoders.0.mlp.0.linear',
        'ddpm.dynamics.encoders.0.mlp.1.linear',
        'ddpm.dynamics.encoders.1.mlp.0.linear',
        'ddpm.dynamics.encoders.1.mlp.1.linear',
        'ddpm.dynamics.encoders.2.mlp.0.linear',
        'ddpm.dynamics.encoders.2.mlp.1.linear',
        'ddpm.dynamics.decoders.0.mlp.0.linear',
        'ddpm.dynamics.decoders.0.mlp.1.linear',
        'ddpm.dynamics.decoders.1.mlp.0.linear',
        'ddpm.dynamics.decoders.1.mlp.1.linear',
        'ddpm.dynamics.decoders.2.mlp.0.linear',
        'ddpm.dynamics.decoders.2.mlp.1.linear',
    ]
    paddle_state_dict = {"state_dict": {}}
    for k in torch_state_dict['state_dict']:
        if "num_batches_tracked" in k:  # 飞桨中无此参数，无需保存
            continue
        v = torch_state_dict['state_dict'][k].detach().cpu().numpy()
        flag = [i in k for i in fc_names]
        if any(flag) and "weight" in k:
            new_shape = [1, 0] + list(range(2, v.ndim))
            print(
                f"name: {k}, ori shape: {v.shape}, new shape: {v.transpose(new_shape).shape}"
            )
            v = v.transpose(new_shape)   # 转置 Linear 层的 weight 参数
        # 将 torch.nn.BatchNorm2d 的参数名称改成 paddle.nn.BatchNorm2D 对应的参数名称
        k = k.replace("running_var", "_variance")
        k = k.replace("running_mean", "_mean")
        # k = k.replace("model.", "")
        # 添加到飞桨权重字典中
        paddle_state_dict['state_dict'][k] = v
    paddle_state_dict["hyper_parameters"] = torch_state_dict["hyper_parameters"]
    paddle.save(paddle_state_dict, paddle_path)


if __name__ == "__main__":
    torch2paddle()
