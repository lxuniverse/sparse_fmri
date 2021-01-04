import torch
import torch.nn as nn
import numpy as np


class MLP_lasso_l0(nn.Module):
    def __init__(self, input_dim, num_classes, layer_dims=(300, 100), loss2lambda=0, r=0,
                 group_num=0, group_method=0, device='cuda'):
        super(MLP_lasso_l0, self).__init__()

        self.device = device

        # network structure
        self.layer_dims = layer_dims
        self.input_dim = input_dim
        layers = []
        for i, dimh in enumerate(self.layer_dims):
            inp_dim = self.input_dim if i == 0 else self.layer_dims[i - 1]
            layers += [nn.Linear(inp_dim, dimh), nn.ReLU()]
        layers.append(nn.Linear(self.layer_dims[-1], num_classes))
        self.output = nn.Sequential(*layers)

        # lasso
        # group_idx = torch.tensor(np.load('/home/xli62/brain/brain_l0_group/group_idx/{}.npy'.format(r)))
        group_idx = torch.tensor(np.load('./data/group_idx_m_{}_r_{}.npy'.format(group_method, r)))
        self.group_idx = group_idx.long() - 1
        self.group_idx = self.group_idx.to(device)

        # for L0 computation
        self.loss2lambda = loss2lambda
        self.loga = torch.randn(group_num, requires_grad=True, device=device)
        self.sig = nn.Sigmoid()
        self.hardtanh = nn.Hardtanh(0, 1)
        self.eps = 1e-20
        self.gamma = -0.1
        self.zeta = 1.1
        self.beta = 0.66
        self.const1 = self.beta * np.log(-self.gamma / self.zeta + self.eps)

    def forward(self, x):
        mask_z = self.l0_train(self.loga)
        mask_z2 = mask_z[self.group_idx]
        masked = x * mask_z2
        return self.output(masked)

    def num_0(self):
        mask_z = self.l0_test(self.loga)
        z_np = mask_z.cpu().detach().numpy()
        num_0 = np.count_nonzero(z_np == 0)
        return num_0

    def forward_test(self, x):
        mask_z = self.l0_test(self.loga)
        mask_z2 = mask_z[self.group_idx]
        masked = x * mask_z2
        out = self.output(masked)
        return out

    def loss2(self):
        return self.loss2lambda * torch.mean(self.sig(self.loga - self.const1))

    def l0_train(self, logAlpha):
        U = torch.rand(logAlpha.size())
        U = U.to(self.device)
        s = self.sig((torch.log(U + self.eps) - torch.log(1 - U + self.eps) + logAlpha + self.eps) / self.beta)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        z = self.hardtanh(s_bar)
        return z

    def l0_test(self, logAlpha):
        s = self.sig(logAlpha / self.beta)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        z = self.hardtanh(s_bar)
        return z

class MLP_choose(nn.Module):
    def __init__(self, vec_select_feature, input_dim, num_classes, layer_dims=(300, 100), device = 'cuda'):
        super(MLP_choose, self).__init__()
        self.device = device
        self.layer_dims = layer_dims
        self.input_dim = input_dim
        self.vec_select_feature = vec_select_feature
        self.vec_select_feature.requires_grad = False
        self.vec_select_feature = self.vec_select_feature.to(device)
        layers = []
        for i, dimh in enumerate(self.layer_dims):
            inp_dim = self.input_dim if i == 0 else self.layer_dims[i - 1]
            layers += [nn.Linear(inp_dim, dimh), nn.ReLU()]
        layers.append(nn.Linear(self.layer_dims[-1], num_classes))
        self.output = nn.Sequential(*layers)

        self.sig = nn.Sigmoid()


    def forward(self, x):
        x = x * self.vec_select_feature
        return self.output(x)