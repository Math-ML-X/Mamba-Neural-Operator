import torch
from torch.nn import MSELoss, L1Loss

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use this one
def eval_2d(input, target, metric_list):
    dict = {}
    for metric_name in metric_list:
        if metric_name in ['MSE', 'L2']:
            dict[metric_name] = MSELoss(reduction='mean')(input, target).item()
        elif metric_name in ['MAE', 'L1']:
            dict[metric_name] = L1Loss(reduction='mean')(input, target).item()
        elif metric_name in ['RMSE']:
            dict[metric_name] = metric_func(input, target)[0].item()
        elif metric_name in ['nRMSE']:
            nRMSE = metric_func(input, target)[1].item()
            dict[metric_name] = nRMSE
        elif metric_name in ['Rel_L2_Norm']:
            dict[metric_name] = LpLoss()(input, target).item()
        elif metric_name in ['CSV']:
            dict[metric_name] = metric_func(input, target)[2].item()
        elif metric_name in ['Max']:
            dict[metric_name] = metric_func(input, target)[3].item()
        else:
            raise NotImplementedError

    return dict


class LpLoss(object):
    '''
    loss function with rel/abs Lp loss
    '''
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def eval_2d_time(input, target, metric_list, length=None):
    if length is None:
        length = input.shape[-2]
    else:
        assert length == input.shape[-2]

    dict_for_dict = {}
    for i in range(length):
        input_t = input[..., i, :]
        target_t = target[..., i, :]
        dict_t = eval_2d(input_t, target_t, metric_list)
        dict_for_dict[f't_{i}'] = dict_t

    return dict_for_dict

def eval_sw2d(input, target, metric_list):
    dict = {}
    for metric_name in metric_list:
        if metric_name in ['MSE', 'L2']:
            dict[metric_name] = MSELoss(reduction='mean')(input, target).item()
        elif metric_name in ['MAE', 'L1']:
            dict[metric_name] = L1Loss(reduction='mean')(input, target).item()
        elif metric_name in ['RMSE']:
            dict[metric_name] = metric_func(input, target)[0].item()
        elif metric_name in ['nRMSE']:
            dict[metric_name] = metric_func(input, target)[1].item()
        elif metric_name in ['CSV']:
            dict[metric_name] = metric_func(input, target)[2].item()
        elif metric_name in ['Max']:
            dict[metric_name] = metric_func(input, target)[3].item()
        elif metric_name in ['MAE', 'L1']:
            dict[metric_name] = L1Loss(reduction='mean')(input, target).item()
        else:
            raise NotImplementedError

    return dict

def eval_darcy(input, target, metric_list):
    dict = {}
    for metric_name in metric_list:
        if metric_name in ['MSE', 'L2']:
            dict[metric_name] = MSELoss(reduction='mean')(input, target).item()
        elif metric_name in ['MAE', 'L1']:
            dict[metric_name] = L1Loss(reduction='mean')(input, target).item()
        elif metric_name in ['RMSE']:
            dict[metric_name] = metric_func(input, target)[0].item()
        elif metric_name in ['nRMSE']:
            dict[metric_name] = metric_func(input, target)[1].item()
        elif metric_name in ['CSV']:
            dict[metric_name] = metric_func(input, target)[2].item()
        elif metric_name in ['Max']:
            dict[metric_name] = metric_func(input, target)[3].item()
        else:
            raise NotImplementedError

    return dict


def eval_convection_diffusion(input, target, metric_list):
    dict = {}
    for metric_name in metric_list:
        if metric_name in ['MSE', 'L2']:
            dict[metric_name] = MSELoss(reduction='mean')(input, target).item()
        elif metric_name in ['MAE', 'L1']:
            dict[metric_name] = L1Loss(reduction='mean')(input, target).item()
        elif metric_name in ['RMSE']:
            dict[metric_name] = metric_func(input, target)[0].item()
        elif metric_name in ['nRMSE']:
            dict[metric_name] = metric_func(input, target)[1].item()
        elif metric_name in ['CSV']:
            dict[metric_name] = metric_func(input, target)[2].item()
        elif metric_name in ['Max']:
            dict[metric_name] = metric_func(input, target)[3].item()
        else:
            raise NotImplementedError

    return dict


def eval_incom_ns2d(input, target, metric_list):
    dict = {}
    for metric_name in metric_list:
        if metric_name in ['MSE', 'L2']:
            dict[metric_name] = MSELoss(reduction='mean')(input, target).item()
        elif metric_name in ['MAE', 'L1']:
            dict[metric_name] = L1Loss(reduction='mean')(input, target).item()
        elif metric_name in ['RMSE']:
            dict[metric_name] = metric_func(input, target)[0].item()
        elif metric_name in ['nRMSE']:
            nRMSE = metric_func(input, target)[1].item()
            dict[metric_name] = nRMSE
        elif metric_name in ['Rel_L2_Norm']:
            from utils.util_loss import LpLoss
            dict[metric_name] = LpLoss()(input, target).item()
        elif metric_name in ['CSV']:
            dict[metric_name] = metric_func(input, target)[2].item()
        elif metric_name in ['Max']:
            dict[metric_name] = metric_func(input, target)[3].item()
        else:
            raise NotImplementedError

    return dict

def eval_com_ns2d(input, target, metric_list):
    dict = {}
    for metric_name in metric_list:
        if metric_name in ['MSE', 'L2']:
            dict[metric_name] = MSELoss(reduction='mean')(input, target).item()
        elif metric_name in ['MAE', 'L1']:
            dict[metric_name] = L1Loss(reduction='mean')(input, target).item()
        elif metric_name in ['RMSE']:
            dict[metric_name] = metric_func(input, target)[0].item()
        elif metric_name in ['nRMSE']:
            dict[metric_name] = metric_func(input, target)[1].item()
        elif metric_name in ['CSV']:
            dict[metric_name] = metric_func(input, target)[2].item()
        elif metric_name in ['Max']:
            dict[metric_name] = metric_func(input, target)[3].item()
        else:
            raise NotImplementedError

    return dict


def eval_dr2d(input, target, metric_list):
    dict = {}
    for metric_name in metric_list:
        if metric_name in ['MSE', 'L2']:
            dict[metric_name] = MSELoss(reduction='mean')(input, target).item()
        elif metric_name in ['MAE', 'L1']:
            dict[metric_name] = L1Loss(reduction='mean')(input, target).item()
        elif metric_name in ['RMSE']:
            dict[metric_name] = metric_func(input, target)[0].item()
        elif metric_name in ['nRMSE']:
            dict[metric_name] = metric_func(input, target)[1].item()
        elif metric_name in ['CSV']:
            dict[metric_name] = metric_func(input, target)[2].item()
        elif metric_name in ['Max']:
            dict[metric_name] = metric_func(input, target)[3].item()
        else:
            raise NotImplementedError

    return dict


def eval_sw2d(input, target, metric_list):
    dict = {}
    for metric_name in metric_list:
        if metric_name in ['MSE', 'L2']:
            dict[metric_name] = MSELoss(reduction='mean')(input, target).item()
        elif metric_name in ['MAE', 'L1']:
            dict[metric_name] = L1Loss(reduction='mean')(input, target).item()
        elif metric_name in ['RMSE']:
            dict[metric_name] = metric_func(input, target)[0].item()
        elif metric_name in ['nRMSE']:
            dict[metric_name] = metric_func(input, target)[1].item()
        elif metric_name in ['CSV']:
            dict[metric_name] = metric_func(input, target)[2].item()
        elif metric_name in ['Max']:
            dict[metric_name] = metric_func(input, target)[3].item()
        else:
            raise NotImplementedError

    return dict


def metric_func(pred, target, if_mean=True, device='cpu'):
    """
    code for calculate metrics discussed in the Brain-storming session
    RMSE, normalized RMSE, max error, RMSE at the boundaries, conserved variables, RMSE in Fourier space, temporal sensitivity
    """
    pred, target = pred.to(device), target.to(device)
    # (batch, nx^i..., timesteps, nc)
    idxs = target.size()
    if len(idxs) == 4:
        pred = pred.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)
    if len(idxs) == 5:
        pred = pred.permute(0, 4, 1, 2, 3)
        target = target.permute(0, 4, 1, 2, 3)
    elif len(idxs) == 6:
        pred = pred.permute(0, 5, 1, 2, 3, 4)
        target = target.permute(0, 5, 1, 2, 3, 4)
    idxs = target.size()
    nb, nc, nt = idxs[0], idxs[1], idxs[-1]

    # RMSE
    err_mean = torch.sqrt(torch.mean((pred.view([nb, nc, -1, nt]) - target.view([nb, nc, -1, nt])) ** 2, dim=2))
    err_RMSE = torch.mean(err_mean, axis=0)
    nrm = torch.sqrt(torch.mean(target.view([nb, nc, -1, nt]) ** 2, dim=2))
    err_nRMSE = torch.mean(err_mean / (nrm + 1e-9), dim=0)

    err_CSV = torch.sqrt(torch.mean(
        (torch.sum(pred.view([nb, nc, -1, nt]), dim=2) - torch.sum(target.view([nb, nc, -1, nt]), dim=2)) ** 2,
        dim=0))
    if len(idxs) == 4:
        nx = idxs[2]
        err_CSV /= nx
    elif len(idxs) == 5:
        nx, ny = idxs[2:4]
        err_CSV /= nx * ny
    elif len(idxs) == 6:
        nx, ny, nz = idxs[2:5]
        err_CSV /= nx * ny * nz
    # worst case in all the data
    err_Max = torch.max(torch.max(
        torch.abs(pred.view([nb, nc, -1, nt]) - target.view([nb, nc, -1, nt])), dim=2)[0], dim=0)[0]

    if if_mean:
        return torch.mean(err_RMSE, dim=[0, -1]), \
               torch.mean(err_nRMSE, dim=[0, -1]), \
               torch.mean(err_CSV, dim=[0, -1]), \
               torch.mean(err_Max, dim=[0, -1])
    else:
        return err_RMSE, err_nRMSE, err_CSV, err_Max