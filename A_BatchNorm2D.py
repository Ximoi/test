import numpy as np

'''
知识点：
1.BN的作用



2.BN如何解决梯度消失



3.BN如何解决梯度爆炸




4.BN的反向传播


常用的超参设置是多少？？？



'''
class BatchNorm2D(object):
    def __init__(self, in_channels, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.state = 'train'

        self.in_channels = in_channels

        # learned parameters
        self.gamma = np.random.rand(1, self.in_channels, 1, 1)
        self.beta = np.random.rand(1, self.in_channels, 1, 1)

        # collected while training
        self.running_mean = None
        self.running_var = None

    def set_state(self, state):
        assert state in ['train', 'val']
        self.state = state

    def __call__(self, input_x):
        # b, c, h, w = input_x.shape

        if self.state == 'train':
            # calculate mean, var
            mean_ = input_x.transpose((1, 0, 2, 3)).reshape((self.in_channels, -1)).mean(axis=1).reshape((1, self.in_channels, 1, 1))

            var_ = input_x.transpose((1, 0, 2, 3)).reshape((self.in_channels, -1)).var(axis=1).reshape((1, self.in_channels, 1, 1))


            # normalize
            x = (input_x - mean_) / np.sqrt(var_ + self.eps)

            # scale and shift
            x = self.gamma * x + self.beta

            # update running_mean, running_var
            if self.running_mean is None:
                self.running_mean = mean_
                self.running_var = var_
            else:
                self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mean_
                self.running_var = self.momentum * self.running_var + (1-self.momentum) * var_

        else:
            # normalize
            x = (input_x - self.running_mean) / np.sqrt(self.running_var + self.eps)

            # scale and shift
            x = self.gamma * x + self.beta

        return x











#
#
# class BatchNorm2D(object):
#     def __init__(self, in_channel):
#
#         self.eps = 1e-5
#         self.state = 'train'
#
#         self.momentum = 0.1
#         self.params = {
#             # 'running_mean': np.zeros(in_channel, dtype=np.float),
#             # 'running_var': np.zeros(in_channel, dtype=np.float),
#             # 'gamma': np.zeros(in_channel, dtype=np.float),
#             # 'beta': np.zeros(in_channel, dtype=np.float),
#         }
#
#     def change_state(self, state):
#         self.state = state
#
#     def __call__(self, input_feat):
#         if self.state == 'train':
#
#             if 'running_mean' not in self.params.keys():
#                 b, c, h, w = input_feat.shape  # batch, channel, height, width
#                 self.params['running_mean'] = np.zeros((c, h, w), dtype=np.float)
#                 self.params['running_var'] = np.ones((c, h, w), dtype=np.float)
#                 self.params['gamma'] = np.ones((c, h, w), dtype=np.float)  #  * 0.1
#                 self.params['beta'] = np.zeros((c, h, w), dtype=np.float)  #  + 1
#
#             u = np.mean(input_feat, axis=0)  # 对batch平均，每个通道一个值
#             v = np.var(input_feat, axis=0)
#
#             normalized_ = (input_feat - u) / np.sqrt(v + self.eps)
#             scale_shifted_ = self.params['gamma'] * normalized_ + self.params['beta']
#
#             # 更新running_mean running_var
#             self.params['running_mean'] = self.momentum*self.params['running_mean'] + (1-self.momentum)*u
#             self.params['running_var'] = self.momentum*self.params['running_var'] + (1-self.momentum)*v
#
#         else:
#
#             normalized_ = (input_feat - self.params['running_mean']) / np.sqrt(self.params['running_val'] + self.eps)
#             scale_shifted_ = self.params['gamma'] * normalized_ + self.params['beta']
#
#         return scale_shifted_


if __name__=='__main__':
    inputa_ = np.random.rand(4, 8, 16, 16).astype('float')
    # print(inputa_)

    bn_layer = BatchNorm2D(in_channels=8, eps=1e-5, momentum=0.1)
    outputa_ = bn_layer(inputa_)

    # print(outputa_)