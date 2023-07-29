"""
Code adopted from AdaATTN:
https://github.com/Huage001/AdaAttN
"""
import torch

class CalcContentReltLoss():
    """Calc Content Relt Loss.
    """
    def __init__(self):
        super(CalcContentReltLoss, self).__init__()

    def __call__(self, pred1, target1, pred2, target2):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        dM = 1.
        Mx = calc_emd_loss(pred1, target1) # pred ，target
        Mx = Mx / (Mx.sum(1, keepdim=True))
        My = calc_emd_loss(pred2, target2) #  pred ，target
        My = My / (My.sum(1, keepdim=True))
        loss_content = torch.abs(
            dM * (Mx - My)).mean() * pred1.shape[2] * pred1.shape[3]
        return loss_content


def calc_emd_loss(pred, target):
    """calculate emd loss.

    Args:
        pred (Tensor): of shape (N, C, H, W). Predicted tensor.
        target (Tensor): of shape (N, C, H, W). Ground truth tensor.
    """
    b, _, h, w = pred.shape
    pred = pred.reshape([b, -1, w * h])
    pred_norm = torch.sqrt((pred**2).sum(1).reshape([b, -1, 1]))
    pred = pred.transpose(2, 1)
    target_t = target.reshape([b, -1, w * h])
    target_norm = torch.sqrt((target**2).sum(1).reshape([b, 1, -1]))
    similarity = torch.bmm(pred, target_t) / pred_norm / target_norm
    dist = 1. - similarity
    return dist
