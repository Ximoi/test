import numpy as np


'''
    np.pad函数的使用？np.pad(x, pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0.0)
    img2col的实现？   .flatten()
    batch样本可否并行？   可以，利用 feat_matrix 和 weight_matrix 做dot时broadcast的机制
'''
# 只有batchsize需要循环
class Conv2D(object):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        # settings
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias


        # 卷积的权重
        self.weights = np.random.rand(out_channels, in_channels, kernel_size, kernel_size).astype(np.float)
        # self.weights = np.ones((out_channels, in_channels, kernel_size, kernel_size)).astype(np.float)
        if self.use_bias:
            self.bias = np.random.rand(out_channels).astype(np.float)
            # self.bias = np.ones(out_channels).astype(np.float)
        else:
            self.bias = None


    def __call__(self, input_feat):
        b, c, h, w = input_feat.shape

        # kernel展开
        weight_matrix = self._kernel2col()

        # padding
        # print(input_feat.shape)
        padded_input_feat = np.pad(input_feat, pad_width=((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant', constant_values=0.0)  #
        # padded_input_feat = input_feat.astype(np.float)
        # print(padded_input_feat.shape)

        self.out_h = (h + 2*self.padding - self.kernel_size) // self.stride + 1
        self.out_w = (w + 2*self.padding - self.kernel_size) // self.stride + 1
        # print(self.out_h, self.out_w)  #

        ''' 对样本进行循环 '''
        # out_lis = []
        # for batch_no in range(b):
        #     # print(padded_input_feat[batch_no, :, :, :])
        #     feat_matrix = self._img2col(padded_input_feat[batch_no, :, :, :])
        #     output_for_one_sample = np.dot(feat_matrix, weight_matrix)  # out_h*out_w行， out_channel列
        #     output_for_one_sample = output_for_one_sample.transpose((1, 0))  # out_channel行， out_h*out_w列
        #     output_for_one_sample = output_for_one_sample.reshape((self.out_channels, self.out_h, self.out_w))
        #     output_for_one_sample = np.expand_dims(output_for_one_sample, axis=0)
        #     out_lis.append(output_for_one_sample)
        # output = np.concatenate(out_lis, axis=0)

        ''' 不对样本进行循环 '''
        feat_matrix_lis = []
        for batch_no in range(b):
            # print(padded_input_feat[batch_no, :, :, :])
            feat_matrix_for_one_sample = self._img2col(padded_input_feat[batch_no, :, :, :])
            feat_matrix_for_one_sample = np.expand_dims(feat_matrix_for_one_sample, axis=0)
            feat_matrix_lis.append(feat_matrix_for_one_sample)

        feat_matrix = np.concatenate(feat_matrix_lis, axis=0)
        output = np.dot(feat_matrix, weight_matrix)  # batchsize, out_h*out_w行， out_channel列
        output = output.transpose((0, 2, 1)).reshape((b, self.out_channels, self.out_h, self.out_w))




            # output_for_one_sample = np.dot(feat_matrix, weight_matrix)  # out_h*out_w行， out_channel列
            # output_for_one_sample = output_for_one_sample.transpose((1, 0))  # out_channel行， out_h*out_w列
            # output_for_one_sample = output_for_one_sample.reshape((self.out_channels, self.out_h, self.out_w))
            # output_for_one_sample = np.expand_dims(output_for_one_sample, axis=0)
            # out_lis.append(output_for_one_sample)




        if self.use_bias:
            temp_bias = np.expand_dims(self.bias, axis=0)
            temp_bias = np.expand_dims(temp_bias, axis=-1)
            temp_bias = np.expand_dims(temp_bias, axis=-1)
            # print(temp_bias.shape, '00000')
            output += temp_bias  # b, out_channels, out_h, out_w    和  out_channels

        return output



    def _img2col(self, input_x):
        c, h, w = input_x.shape
        # print(c, h, w)

        feat_mat_height = self.out_h * self.out_w
        feat_mat_width = self.kernel_size * self.kernel_size * c
        # print(feat_mat_height)
        # print(feat_mat_width)
        feat_matrix = np.empty(shape=(feat_mat_height, feat_mat_width), dtype=np.float)
        for m in range(self.out_h):
            for n in range(self.out_w):
                temp = input_x[:, m*self.stride:m*self.stride+self.kernel_size, n*self.stride:n*self.stride+self.kernel_size].flatten()  # channel height width
                # print(m, n, temp)
                feat_matrix[m * self.out_w + n, :] = temp

        return feat_matrix


    def _kernel2col(self):
        out_channel, in_channel, kernel_size, kernel_size = self.weights.shape  # out_channel, in_channel, kernel_size, kernel_size

        weight_matrix = np.empty((in_channel*kernel_size*kernel_size, out_channel)).astype(np.float)
        for m in range(out_channel):
            for n in range(in_channel):
                temp = self.weights[m, n, :, :].flatten()
                weight_matrix[n*self.kernel_size*self.kernel_size:(n+1)*self.kernel_size*self.kernel_size, m] = temp

        return weight_matrix


if __name__=='__main__':
    b, c, h, w = 2, 1, 4, 4
    inputa_ = np.array(range(b*c*h*w)).reshape((b, c, h, w)).astype(np.float) + 1
    print(inputa_)
    print(inputa_.shape)

    # in_channels, out_channels, kernel_size, stride, padding
    conv_layer = Conv2D(c, 2, kernel_size=3, stride=1, padding=1, bias=True)
    out_ = conv_layer(inputa_)
    print(out_)
    print(out_.shape)






