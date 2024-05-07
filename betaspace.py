import torch

def betaspace_F(data, x):
    device = data.device

    # 按列求和，求入度
    SAout = torch.sum(data, dim=0)

    x_w = data * x

    # 网络结构没有连通度
    if torch.sum(SAout) == 0:
        β_eff = torch.tensor(0, dtype=torch.float, device=device)
        x_eff = torch.tensor(0, dtype=torch.float, device=device)
    else:
        # 整个网络
        β_eff = torch.sum(torch.sum(torch.matmul(data, data), dim=0)) / torch.sum(SAout)
        # 入度/出度
        x_eff = torch.sum(x_w) / torch.sum(SAout)

    return β_eff, x_eff