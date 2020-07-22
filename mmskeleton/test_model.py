import torch
from mmskeleton.utils.config import Config
from mmskeleton.models.backbones.st_gcn_aaai18 import ST_GCN_18

if __name__ == "__main__":
    cfg = Config.fromfile(
        '/media/F/thesis/libs/mmskeleton/configs/recognition/st_gcn_aaai18/kinetics-skeleton/test.yaml')

    mcfg = dict(layout='openpose',
                strategy='uniform',
                max_hop=2,
                dilation=1)

    count_n_params = lambda model_: sum(p.numel() for p in model_.parameters())
    model = ST_GCN_18(in_channels=3, graph_cfg=mcfg)
    print(count_n_params(model) / 1e6)
    # N, C, T, V, M = x.size()
    x = torch.rand(16, 3, 27, 18)
    preds = model(x)
    print(preds.shape)
