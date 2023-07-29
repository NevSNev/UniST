import os
import matplotlib
import matplotlib.pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
matplotlib.use('Agg')
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import common, train_utils
from configs import data_configs
from datasets.video_image_dataset import Video_image_Dataset
from datasets.style_dataset import StyleDataset
from models.model import video_Style_transfer
from training.ranger import Ranger
import torch.nn.functional as F
from criteria.vgg import encoder5 as loss_network
from criteria.critertion import LossCriterion
from criteria.critertion_content import LossCriterion_content
from criteria.cos_loss import  CalcContentReltLoss

class Coach:
    def __init__(self, opts):
        self.opts = opts
        self.global_step = 0
        self.device = 'cuda:0'
        self.opts.device = self.device

        # Initialize network
        self.net = video_Style_transfer(self.opts).to(self.device)
        self.vgg = loss_network().to(self.device)
        self.vgg.load_state_dict(torch.load('weight/vgg_r51.pth'))
        self.cos_loss = CalcContentReltLoss()

        for param in self.vgg.parameters():
            param.requires_grad = False
        for param in self.net.model.encoder.parameters():
            param.requires_grad = False

        arg1 = ['r11','r21','r31','r41']
        arg2 = ['r41']
        self.styleloss =LossCriterion(arg1,arg1,self.opts.style_lambda,self.opts.content_lambda).to(self.device)
        self.IDloss = LossCriterion_content(arg1).to(self.device)

        # Initialize optimizer
        self.optimizer = self.configure_optimizers()

        # Initialize dataset
        self.train_content_frames,self.test_content_frames ,self.train_style ,self.test_style = self.configure_datasets()

        self.train_content_dataloader = DataLoader(self.train_content_frames,
                                                   batch_size=self.opts.batch_size,
                                                   shuffle=True,
                                                   num_workers=int(self.opts.workers),
                                                   drop_last=True)

        self.train_style_dataloader = DataLoader(self.train_style,
                                                 batch_size=self.opts.batch_size,
                                                 shuffle=True,
                                                 num_workers=int(self.opts.workers),
                                                 drop_last=True)

        self.test_content_dataloader = DataLoader(self.test_content_frames,
                                                  batch_size=self.opts.test_batch_size,
                                                  shuffle=False,
                                                  num_workers=int(self.opts.test_workers),
                                                  drop_last=True)

        self.test_style_dataloader = DataLoader(self.test_style,
                                                batch_size=self.opts.test_batch_size,
                                                shuffle=False,
                                                num_workers=int(self.opts.test_workers),
                                                drop_last=True)

        # Initialize logger
        log_dir = os.path.join(opts.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=log_dir)

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None
        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps

    def train(self):
        self.net.train()

        while self.global_step < self.opts.max_steps:
            for batch_idx, (content_frames,style) in enumerate(zip(self.train_content_dataloader,self.train_style_dataloader)):
                self.optimizer.zero_grad()
                x = content_frames
                b,t,_,_,_ = content_frames.size()
                y = style

                x, y = x.to(self.device).float(), y.to(self.device).float()
                transfer_result = self.net.forward(x,y,id_loss='transfer')
                _,c,h,w = transfer_result.size()
                transfer_result = transfer_result.view(b, t, c, h, w)

                y_style = self.net.forward(y,y,id_loss='style')
                y_content = self.net.forward(x,x,id_loss='content').view(b, t, c, h, w)

                loss, loss_dict = self.calc_loss(x[:,0,:,:,:], y, transfer_result[:,0,:,:,:],y_content[:,0,:,:,:],y_style)
                loss_dict['temporary_loss'] = 0.0
                loss.backward(retain_graph=True)
                for i in range(1,t):
                    loss_, loss_dic_ = self.calc_loss(x[:,i,:,:,:], y, transfer_result[:,i,:,:,:],y_content[:,i,:,:,:],y_style)
                    if i > 2:

                        loss_, loss_dic_ = self.cal_temporary_loss(x[:,i-1,:,:,:],x[:,i,:,:,:],transfer_result[:,i-1,:,:,:],transfer_result[:,i,:,:,:],loss_, loss_dic_)

                    if i == t-1:
                        loss_.backward()
                    else :
                        loss_.backward(retain_graph=True)
                    for k,v in loss_dic_.items():
                        loss_dict[k] += v
                self.optimizer.step()

                # Logging related
                if self.global_step % self.opts.image_interval == 0 or (self.global_step < 1000 and self.global_step % 25 == 0):
                    self.parse_and_log_images( x[0,:,:,:,:], y[0], transfer_result[0,:,:,:,:], title='images/train/result')
                if self.global_step % self.opts.board_interval == 0:
                    self.print_metrics(loss_dict, prefix='train')
                    self.log_metrics(loss_dict, prefix='train')

                # Validation related
                val_loss_dict = None
                if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
                    val_loss_dict = self.validate()
                    if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
                        self.best_val_loss = val_loss_dict['loss']
                        self.checkpoint_me(val_loss_dict, is_best=True)

                if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
                    if val_loss_dict is not None:
                        self.checkpoint_me(val_loss_dict, is_best=False)
                    else:
                        self.checkpoint_me(loss_dict, is_best=False)

                if self.global_step == self.opts.max_steps:
                    print('OMG, finished training!')
                    break

                self.global_step += 1

    def validate(self):

        self.net.eval()

        agg_loss_dict = []
        for batch_idx,(content_frames,style) in enumerate(zip(self.test_content_dataloader,self.test_style_dataloader)):
            x = content_frames
            b,t,_,_,_ = content_frames.size()
            y = style

            with torch.no_grad():
                x, y = x.to(self.device).float(), y.to(self.device).float()
                transfer_result  = self.net.forward(x,y,id_loss='transfer')
                _,c,h,w = transfer_result.size()
                y_style = self.net.forward(y,y,id_loss='style')
                transfer_result = transfer_result.view(b, t, c, h, w)
                y_content = self.net.forward(x,x,id_loss='content').view(b, t, c, h, w)

                loss, cur_loss_dict = self.calc_loss(x[:,0,:,:,:], y, transfer_result[:,0,:,:,:],y_content[:,0,:,:,:],y_style)
                cur_loss_dict['temporary_loss'] = 0.0
                for i in range(1,t):
                    loss_, loss_dic_ = self.calc_loss(x[:,i,:,:,:], y, transfer_result[:,i,:,:,:],y_content[:,i,:,:,:],y_style)
                    if i > 2:
                        loss_, loss_dic_ = self.cal_temporary_loss(x[:,i-1,:,:,:],x[:,i,:,:,:],transfer_result[:,i-1,:,:,:],transfer_result[:,i,:,:,:],loss_, loss_dic_)
                    for k,v in loss_dic_.items():
                        cur_loss_dict[k] += v

            agg_loss_dict.append(cur_loss_dict)
            self.parse_and_log_images( x[0,:,:,:,:], y[0], transfer_result[0,:,:,:,:], title='images/test/result',subscript='{:04d}'.format(batch_idx))

            if self.global_step == 0 and batch_idx >= 4:
                self.net.train()
                return None

        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')

        self.net.train()
        return loss_dict

    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else 'iteration_{}.pt'.format(self.global_step)
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write('**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
            else:
                f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))

    def configure_optimizers(self):
        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.model.parameters()), lr=self.opts.learning_rate)
        else:
            optimizer = Ranger(filter(lambda p: p.requires_grad, self.net.model.parameters()), lr=self.opts.learning_rate)
        return optimizer

    def configure_datasets(self):
        if self.opts.dataset_type not in data_configs.DATASETS.keys():
            Exception('{} is not a valid dataset_type'.format(self.opts.dataset_type))
        print('Loading dataset for {}'.format('video_style_transfer'))
        dataset_args = data_configs.DATASETS['video_style_transfer']
        transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
        train_content_frames =Video_image_Dataset(video_data_root=self.opts.video_path_train,image_data_root=self.opts.image_path_train,image_transform=transforms_dict)
        test_content_frames = Video_image_Dataset(video_data_root=self.opts.video_path_test,image_data_root=self.opts.image_path_test,image_transform=transforms_dict)
        train_style = StyleDataset(style_root=self.opts.style_path_train,
                                   style_transform=transforms_dict,
                                   opts=self.opts)
        test_style = StyleDataset(style_root=self.opts.style_path_test,
                                  style_transform=transforms_dict,
                                  opts=self.opts)

        print("Number of training samples: {}".format(len(train_content_frames)))
        print("Number of test samples: {}".format(len(test_content_frames)))
        print("Number of training samples: {}".format(len(train_style)))
        print("Number of test samples: {}".format(len(test_style)))
        return train_content_frames, test_content_frames ,train_style ,test_style

    def calc_loss(self, x, y, y_hat , y_content,y_style):
        loss_dict = {}

        sF_loss = self.vgg(y)
        cF_loss = self.vgg(x)
        tF = self.vgg(y_hat)
        loss,styleLoss,contentLoss = self.styleloss(tF,sF_loss,cF_loss)
        loss_dict['contentLoss'] = float(contentLoss)
        loss_dict['styleLoss'] = float(styleLoss)

        id_1_loss = F.mse_loss(x, y_content) +F.mse_loss(y, y_style)
        id_1_loss = id_1_loss * self.opts.id_1_lambda

        tf_content = self.vgg(y_content)
        contentLoss1 = self.IDloss(tf_content,cF_loss)
        tf_style = self.vgg(y_style)
        contentLoss2 = self.IDloss(tf_style,sF_loss)
        id_2_loss =(contentLoss1+contentLoss2) * self.opts.id_2_lambda

        loss_dict['id_1_loss'] = float(id_1_loss)
        loss_dict['id_2_loss'] = float(id_2_loss)

        loss = id_2_loss + id_1_loss + loss
        loss_dict['loss'] = float(loss)
        return loss, loss_dict

    def cal_temporary_loss(self,x_former,x_latter,y_hat_former,y_hat_latter,loss,loss_dict):
        cF_former = self.vgg(x_former)
        cF_latter = self.vgg(x_latter)
        tF_former = self.vgg(y_hat_former)
        tF_latter = self.vgg(y_hat_latter)
        temporary_loss = self.cos_loss(cF_former['r31'], cF_latter['r31'], tF_former['r31'], tF_latter['r31']) + \
                         self.cos_loss(cF_former['r41'], cF_latter['r41'], tF_former['r41'], tF_latter['r41'])
        temporary_loss = temporary_loss * self.opts.temporary_lambda
        loss_dict.update({'temporary_loss':float(temporary_loss)})

        loss = loss + temporary_loss
        loss_dict['loss'] = float(loss)
        return loss,loss_dict


    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar('{}/{}'.format(prefix, key), value, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print('Metrics for {}, step {}'.format(prefix, self.global_step))
        for key, value in metrics_dict.items():
            print('\t{} = '.format(key), value)

    def parse_and_log_images(self, x, y, y_hat, title, subscript=None, display_count=3):
        im_data = []
        for i in range(display_count):
            cur_im_data = {
                'input': common.log_input_image(x[i], self.opts),
                'target': common.tensor2im(y),
                'output': common.tensor2im(y_hat[i]),
            }

            im_data.append(cur_im_data)
        self.log_images(title, im_data=im_data, subscript=subscript)

    def log_images(self, name, im_data, subscript=None, log_latest=False):
        fig = common.vis_faces(im_data)
        step = self.global_step
        if log_latest:
            step = 0
        if subscript:
            path = os.path.join(self.logger.log_dir, name, '{}_{:04d}.jpg'.format(subscript, step))
        else:
            path = os.path.join(self.logger.log_dir, name, '{:04d}.jpg'.format(step))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)

    def __get_save_dict(self):
        save_dict = {
            'state_dict': self.net.state_dict(),
            'opts': vars(self.opts)
        }

        return save_dict