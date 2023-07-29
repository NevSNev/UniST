import matplotlib


matplotlib.use('Agg')
import torch
from torch import nn
from models.transformer.Models import Transformer

def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


class video_Style_transfer(nn.Module):

    def __init__(self, opts):
        super(video_Style_transfer, self).__init__()
        self.opts = opts #
        self.model = Transformer()
        self.load_weights()

    def load_weights(self):
        print('Loading model from checkpoint')
        ckpt = torch.load('weight/UniST_model.pt', map_location='cpu')
        self.model.load_state_dict(get_keys(ckpt, 'model'),strict=False)

    def forward(self, content_frames,style_images, id_loss='transfer', tab = None):
        transfer_result = self.model(content_frames , style_images, id_loss, tab)
        return transfer_result



