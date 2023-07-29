import torch
import torch.nn as nn

class LossCriterion_content(nn.Module):
    def __init__(self,content_layers):
        super(LossCriterion_content,self).__init__()
        self.content_layers = content_layers
        self.contentLosses = [nn.MSELoss()] * len(content_layers)

    def forward(self,tF,cF):
        # content loss
        totalContentLoss = 0
        for i,layer in enumerate(self.content_layers):
            cf_i = cF[layer]
            cf_i = cf_i.detach()
            tf_i = tF[layer]
            loss_i = self.contentLosses[i]
            totalContentLoss += loss_i(tf_i,cf_i)

        return totalContentLoss
