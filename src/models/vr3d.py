import torch
import torch.nn as nn
import torch.nn.functional as F 

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        
    def forward(self, targets, logits):
        batch_size = targets.size(0)
        logits = logits.view(batch_size, -1).type(torch.FloatTensor)
        targets = targets.view(batch_size, -1).type(torch.FloatTensor)
        
        # compute MSE
        l2_norm = torch.linalg.norm(targets-logits) / batch_size
        
        # return the loss
        return l2_norm
		
class LidarObjectDetection_CNN(nn.Module):
    def __init__(self, in_channels, obj_label_len, base_filters=8, n_blocks=6, depth_latent_vector_len=8192):
        super(LidarObjectDetection_CNN, self).__init__()

        # define layers
        self.in_channels = in_channels
        self.base_filters = base_filters		
        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.6)
        self.obj_label_len = obj_label_len

        ## Convolution blocks
        self.layers = {}
        # Block 1
        self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3d_c1_2 = nn.Conv3d(self.base_filters, self.base_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.lrelu_conv_c1 = self.lrelu_conv(self.base_filters, self.base_filters)
        self.inorm3d_c1 = nn.InstanceNorm3d(self.base_filters)		
        # Block 2
        self.conv3d_c2 = nn.Conv3d(self.base_filters, self.base_filters*2, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_filters*2, self.base_filters*2)
        self.inorm3d_c2 = nn.InstanceNorm3d(self.base_filters*2)		
        # Block 3
        self.conv3d_c3 = nn.Conv3d(self.base_filters*2, self.base_filters*4, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_filters*4, self.base_filters*4)
        self.inorm3d_c3 = nn.InstanceNorm3d(self.base_filters*4)		
        # Block 4
        self.conv3d_c4 = nn.Conv3d(self.base_filters*4, self.base_filters*8, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_filters*8, self.base_filters*8)
        self.inorm3d_c4 = nn.InstanceNorm3d(self.base_filters*8)		
        # Block 5
        self.conv3d_c5 = nn.Conv3d(self.base_filters*8, self.base_filters*16, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_filters*16, self.base_filters*16)
        self.inorm3d_c5 = nn.InstanceNorm3d(self.base_filters*16)		
        # Block 6
        self.conv3d_c6 = nn.Conv3d(self.base_filters*16, self.base_filters*32, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm_lrelu_conv_c6 = self.norm_lrelu_conv(self.base_filters*32, self.base_filters*32)
        self.inorm3d_c6 = nn.InstanceNorm3d(self.base_filters*32)	
        # Block 7
        self.conv3d_c7 = nn.Conv3d(self.base_filters*32, self.obj_label_len, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm_lrelu_conv_c7 = self.norm_lrelu_conv(self.obj_label_len, self.obj_label_len)
        # self.maxpool_c7 = nn.MaxPool3d((1, 1, 1), stride=(2, 1, 1))
        self.inorm3d_c7 = nn.InstanceNorm3d(self.obj_label_len)	


    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        ## Block 1
        out = self.conv3d_c1_1(x)
        residual_1 = out
        out = self.lrelu(out)
        out = self.conv3d_c1_2(out)
        out = self.dropout3d(out)
        out = self.lrelu_conv_c1(out)
        # Element Wise Summation
        out += residual_1
        out = self.inorm3d_c1(out)
        out = self.lrelu(out)

        ## Block 2
        out = self.conv3d_c2(out)
        residual_2 = out
        out = self.norm_lrelu_conv_c2(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c2(out)
        out += residual_2
        out = self.inorm3d_c2(out)
        out = self.lrelu(out)

        ## Block 3
        out = self.conv3d_c3(out)
        residual_3 = out
        out = self.norm_lrelu_conv_c3(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c3(out)
        out += residual_3
        out = self.inorm3d_c3(out)
        out = self.lrelu(out)

        ## Block 4
        out = self.conv3d_c4(out)
        residual_4 = out
        out = self.norm_lrelu_conv_c4(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c4(out)
        out += residual_4
        out = self.inorm3d_c4(out)
        out = self.lrelu(out)

        ## Block 5
        out = self.conv3d_c5(out)
        residual_5 = out
        out = self.norm_lrelu_conv_c5(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c5(out)
        out += residual_5
        out = self.inorm3d_c5(out)
        out = self.lrelu(out)

        ## Block 6
        out = self.conv3d_c6(out)
        residual_6 = out
        out = self.norm_lrelu_conv_c6(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c6(out)
        out += residual_6
        out = self.inorm3d_c6(out)
        out = self.lrelu(out)

        ## Block 7
        out = self.conv3d_c7(out)
        residual_7 = out
        out = self.norm_lrelu_conv_c7(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c7(out)
        out += residual_7
        out = self.inorm3d_c7(out)
        out = self.lrelu(out)
        # out = self.maxpool_c7(out)

        ## Fully-connected layers
        out = out.permute(0, 3, 4, 1, 2)
        out = out.contiguous().view(-1, out.shape[1]*out.shape[2]*out.shape[3]*out.shape[4])
        return out

class LidarObjectDetection_FC(nn.Module):
    def __init__(self, n_xgrids, n_ygrids, in_dim, obj_label_len):
        super(LidarObjectDetection_FC, self).__init__()

        self.n_xgrids = n_xgrids
        self.n_ygrids = n_ygrids
        self.in_dim = in_dim
        self.obj_label_len = obj_label_len
        self.out_channels = n_xgrids * n_ygrids * obj_label_len

        # ## Fully-Connected Layers	
        # First fully connected layer
        self.fc1 = nn.Linear(self.in_dim, 4096)
        self.fc2 = nn.Linear(4096, self.out_channels)

        # activation functions
        self.sigmoid = torch.nn.Sigmoid()	
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = out.view(-1, self.n_xgrids, self.n_ygrids, self.obj_label_len)
        conf_out = self.sigmoid(out[:,:,:,0]).unsqueeze(-1)
        pose_out = out[:,:,:,1:9]
        class_out = self.softmax(out[:,:,:,9:])
        out = torch.cat((conf_out, pose_out, class_out), dim=-1).contiguous().view(-1,self.n_xgrids*self.n_ygrids*self.obj_label_len)
        return out

# class EncoderBlock(nn.Module):

#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, normalization=True, norm_type='instance_norm'):

#         super(EncoderBlock, self).__init__() 

#         self.convA = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
#                             kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
#         self.convB = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1)
#         self.leakyrelu = nn.LeakyReLU(0.2)
#         self.instance_norm = nn.InstanceNorm2d(out_channels)
#         self.batch_norm = nn.BatchNorm2d(out_channels)

#         self.normalization = normalization
#         self.norm_type = norm_type
    
#     def forward(self, x):

#         x = self.convA(x)
#         x = self.leakyrelu(x)
#         x = self.convB(x)
#         x = self.leakyrelu(x)
#         if self.normalization == True and self.norm_type == 'instance_norm':
#             x = self.instance_norm(x)
#         elif self.normalization == True and self.norm_type == 'batch_norm':
#             x = self.batch_norm(x)
        
#         return x

# class DecoderBlock(nn.Module):

#     def __init__(self, in_channels, out_channels, normalization=True, norm_type='instance_norm'):

#         super(DecoderBlock, self).__init__() 

#         self.in_channels = in_channels
#         self.out_channels = out_channels

#         self.convA = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
#         self.leakyrelu = nn.LeakyReLU(0.2)
#         self.convB = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
#         self.instance_norm = nn.InstanceNorm2d(out_channels)
#         self.batch_norm = nn.BatchNorm2d(out_channels)

#         self.normalization = normalization
#         self.norm_type = norm_type

#     def forward(self, x, concat_with=None):
#         upsampled_x = x
#         if concat_with is not None:
#             concat_h_dim = concat_with.shape[2]
#             concat_w_dim = concat_with.shape[3]

#             upsampled_x = F.interpolate(x, size=[concat_h_dim, concat_w_dim], mode="bilinear", align_corners=True)
#             upsampled_x = torch.cat([upsampled_x, concat_with], dim=1)
        
#         upsampled_x = self.convA(upsampled_x)
#         upsampled_x = self.leakyrelu(upsampled_x)
#         upsampled_x = self.convB(upsampled_x)
#         upsampled_x = self.leakyrelu(upsampled_x)
#         if self.normalization == True and self.norm_type == 'instance_norm':
#             upsampled_x = self.instance_norm(upsampled_x)
#         elif self.normalization == True and self.norm_type == 'batch_norm':
#             upsampled_x = self.batch_norm(upsampled_x)

#         return upsampled_x

class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, normalization=True, norm_type='instance_norm'):

        super(EncoderBlock, self).__init__() 

        self.convA = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                            kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.convB = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.convC = nn.Conv2d(out_channels * 2, out_channels, kernel_size=kernel_size, stride=1, padding=1)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.batch_norm = nn.BatchNorm2d(out_channels)

        self.normalization = normalization
        self.norm_type = norm_type
    
    def forward(self, x):

        x1 = self.convA(x)
        x1 = self.leakyrelu(x1)
        x2 = self.convB(x1)
        x2 = self.leakyrelu(x2)
        x = torch.cat((x1, x2), 1)
        x = self.convC(x)
        x = self.leakyrelu(x)
        if self.normalization == True and self.norm_type == 'instance_norm':
            x = self.instance_norm(x)
        elif self.normalization == True and self.norm_type == 'batch_norm':
            x = self.batch_norm(x)
        
        return x

class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, normalization=True, norm_type='instance_norm'):

        super(DecoderBlock, self).__init__() 

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.convA = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.convC = nn.Conv2d(out_channels * 2, out_channels, 3, 1, 1)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.batch_norm = nn.BatchNorm2d(out_channels)

        self.normalization = normalization
        self.norm_type = norm_type

    def forward(self, x, concat_with=None):
        upsampled_x = x
        if concat_with is not None:
            concat_h_dim = concat_with.shape[2]
            concat_w_dim = concat_with.shape[3]

            upsampled_x = F.interpolate(x, size=[concat_h_dim, concat_w_dim], mode="bilinear", align_corners=True)
            upsampled_x = torch.cat([upsampled_x, concat_with], dim=1)
        
        upsampled_x1 = self.convA(upsampled_x)
        upsampled_x1 = self.leakyrelu(upsampled_x1)
        upsampled_x2 = self.convB(upsampled_x1)
        upsampled_x2 = self.leakyrelu(upsampled_x2)
        upsampled_x = torch.cat((upsampled_x1, upsampled_x2), 1)
        upsampled_x = self.convC(upsampled_x)
        upsampled_x = self.leakyrelu(upsampled_x)
        if self.normalization == True and self.norm_type == 'instance_norm':
            x = self.instance_norm(x)
        elif self.normalization == True and self.norm_type == 'batch_norm':
            x = self.batch_norm(x)

        return upsampled_x

class DepthEncoder(nn.Module):

    def __init__(self):
        super(DepthEncoder, self).__init__() 

        self.conv0 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)

        self.downsample1 = EncoderBlock(in_channels=8, out_channels=16, normalization=False)
        self.downsample2 = EncoderBlock(in_channels=16, out_channels=32)
        self.downsample3 = EncoderBlock(in_channels=32, out_channels=64)
        self.downsample4 = EncoderBlock(in_channels=64, out_channels=128)
        self.downsample5 = EncoderBlock(in_channels=128, out_channels=256)
        self.downsample6 = EncoderBlock(in_channels=256, out_channels=512)
        
        self.flatten = nn.Flatten()

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.downsample1(x0)
        x2 = self.downsample2(x1)
        x3 = self.downsample3(x2)
        x4 = self.downsample4(x3)
        x5 = self.downsample5(x4)
        x6 = self.downsample6(x5)
        x6 = self.flatten(x6)

        return x0, x1, x2, x3, x4, x5, x6

class DepthDecoder(nn.Module):

    def __init__(self, depth_latent_vector_len, combined_latent_vector_len, in_channels=256, out_channels=1):

        super(DepthDecoder, self).__init__()

        self.in_channels = in_channels

        self.fc0 = nn.Linear(combined_latent_vector_len, depth_latent_vector_len)
        self.conv0 = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

        self.upsample1 = DecoderBlock(in_channels + in_channels // 2, in_channels // 2)
        self.upsample2 = DecoderBlock((in_channels // 2) + (in_channels // 4), (in_channels // 4))
        self.upsample3 = DecoderBlock((in_channels // 4) + (in_channels // 8), (in_channels // 8))
        self.upsample4 = DecoderBlock((in_channels // 8) + (in_channels // 16), (in_channels // 16))
        self.upsample5 = DecoderBlock((in_channels // 16) + (in_channels // 32), (in_channels // 32))
        self.upsample6 = DecoderBlock((in_channels // 32) + (in_channels // 64), (in_channels // 64), normalization=False)

        self.conv_final = nn.Conv2d(in_channels // 64, out_channels, 3, 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, features):

        x_block0 = features[0]
        x_block1 = features[1]
        x_block2 = features[2]
        x_block3 = features[3]
        x_block4 = features[4]
        x_block5 = features[5]
        latent_vector = features[6]

        x_block6 = self.fc0(latent_vector)
        x0 = x_block6.view((-1, self.in_channels, x_block5.shape[2]//2, x_block5.shape[3]//2))
        x0 = self.conv0(x0)
        x1 = self.upsample1(x0, x_block5)
        x2 = self.upsample2(x1, x_block4)
        x3 = self.upsample3(x2, x_block3)
        x4 = self.upsample4(x3, x_block2)
        x5 = self.upsample5(x4, x_block1)
        x6 = self.upsample6(x5, x_block0)

        return self.tanh(self.conv_final(x6))

class DenseDepth(nn.Module):

    def __init__(self):
        super(DenseDepth, self).__init__()

        self.encoder = DepthEncoder()
        self.decoder = DepthDecoder()
    
    def forward(self, x):
        x0, x1, x2, x3, x4, x5, x6 = self.encoder(x)
        depth_pred = self.decoder((x0, x1, x2, x3, x4, x5, x6))
        return x6, depth_pred

class VR3Dense(nn.Module):

    def __init__(self, in_channels, n_xgrids, n_ygrids, obj_label_len, base_filters=8, n_blocks=6, dense_depth=True, train_depth_only=False, train_obj_only=False):
        super(VR3Dense, self).__init__()

        self.dense_depth = dense_depth
        self.depth_latent_vector_len = 0
        self.latent_vector_len = (n_xgrids*n_ygrids*obj_label_len)
        self.train_depth_only = train_depth_only
        self.train_obj_only = train_obj_only
        if (train_depth_only == True) and (train_obj_only == True):
            raise Exception('Only one of \'train_depth_only\', \'train_obj_only\' can be set to true at a time.')
        
        self.lidar_obj_fc_in_dim = 16*16*obj_label_len
        if dense_depth:
            in_dim = (512, 256)
            n_downsampling_blocks = 6
            last_block_n_channels = 512
            self.depth_latent_vector_len = (in_dim[0] // (2**n_downsampling_blocks)) * (in_dim[1] // (2**n_downsampling_blocks)) * last_block_n_channels
            self.depth_encoder = DepthEncoder()
            self.depth_decoder = DepthDecoder(self.depth_latent_vector_len, self.depth_latent_vector_len+self.lidar_obj_fc_in_dim, last_block_n_channels)

        self.lidar_object_detection_cnn = LidarObjectDetection_CNN(in_channels, obj_label_len, base_filters, n_blocks, 0)
        self.lidar_object_detection_fc = LidarObjectDetection_FC(n_xgrids, n_ygrids, self.lidar_obj_fc_in_dim, obj_label_len)
    
    def forward(self, x_lidar, x_camera):
        if self.train_depth_only:
            with torch.no_grad():
                latent_vector = self.lidar_object_detection_cnn(x_lidar)
        else:
            latent_vector = self.lidar_object_detection_cnn(x_lidar)
        depth_pred = None

        if self.dense_depth:
            if self.train_obj_only:
                with torch.no_grad():
                    x0, x1, x2, x3, x4, x5, x6 = self.depth_encoder(x_camera)
                    depth_latent_vector = x6
                    depth_latent_vector = torch.cat([latent_vector.detach(), depth_latent_vector], dim=1)
                    depth_pred = self.depth_decoder((x0, x1, x2, x3, x4, x5, depth_latent_vector))
            else:
                x0, x1, x2, x3, x4, x5, x6 = self.depth_encoder(x_camera)
                depth_latent_vector = x6
                depth_latent_vector = torch.cat([latent_vector.detach(), depth_latent_vector], dim=1)
                depth_pred = self.depth_decoder((x0, x1, x2, x3, x4, x5, depth_latent_vector))

        if self.train_depth_only:
            with torch.no_grad():
                object_pose_pred = self.lidar_object_detection_fc(latent_vector)
        else:
            object_pose_pred = self.lidar_object_detection_fc(latent_vector)

        if self.dense_depth:
            return_tuple = (object_pose_pred, depth_pred)
        else:
            return_tuple = object_pose_pred
        return return_tuple

# class VR3Dense(nn.Module):

#     def __init__(self, in_channels, n_xgrids, n_ygrids, obj_label_len, base_filters=8, n_blocks=6, dense_depth=True, train_depth_only=False, train_obj_only=False):
#         super(VR3Dense, self).__init__()

#         self.dense_depth = dense_depth
#         self.depth_latent_vector_len = 0
#         self.latent_vector_len = (n_xgrids*n_ygrids*obj_label_len)
#         self.train_depth_only = train_depth_only
#         self.train_obj_only = train_obj_only
#         if (train_depth_only == True) and (train_obj_only == True):
#             raise Exception('Only one of \'train_depth_only\', \'train_obj_only\' can be set to true at a time.')
        
#         self.lidar_obj_fc_in_dim = 16*16*obj_label_len
#         if dense_depth:
#             in_dim = 256
#             n_downsampling_blocks = 5
#             last_block_n_channels = 256
#             self.depth_latent_vector_len = (in_dim // (2**n_downsampling_blocks)) * (in_dim // (2**n_downsampling_blocks)) * last_block_n_channels
#             self.depth_encoder = DepthEncoder()
#             self.depth_decoder = DepthDecoder(self.depth_latent_vector_len, self.depth_latent_vector_len, last_block_n_channels)

#         self.lidar_object_detection_cnn = LidarObjectDetection_CNN(in_channels, obj_label_len, base_filters, n_blocks, 0)
#         self.lidar_object_detection_fc = LidarObjectDetection_FC(n_xgrids, n_ygrids, self.lidar_obj_fc_in_dim, obj_label_len)
    
#     def forward(self, x_lidar, x_camera):
#         if self.train_depth_only:
#             with torch.no_grad():
#                 latent_vector = self.lidar_object_detection_cnn(x_lidar)
#         else:
#             latent_vector = self.lidar_object_detection_cnn(x_lidar)
#         depth_pred = None

#         if self.dense_depth:
#             if self.train_obj_only:
#                 with torch.no_grad():
#                     x0, x1, x2, x3, x4, x5 = self.depth_encoder(x_camera)
#                     depth_latent_vector = x5
#                     depth_pred = self.depth_decoder((x0, x1, x2, x3, x4, depth_latent_vector))
#             else:
#                 x0, x1, x2, x3, x4, x5 = self.depth_encoder(x_camera)
#                 depth_latent_vector = x5
#                 depth_pred = self.depth_decoder((x0, x1, x2, x3, x4, depth_latent_vector))

#         if self.train_depth_only:
#             with torch.no_grad():
#                 object_pose_pred = self.lidar_object_detection_fc(latent_vector)
#         else:
#             object_pose_pred = self.lidar_object_detection_fc(latent_vector)

#         if self.dense_depth:
#             return_tuple = (object_pose_pred, depth_pred)
#         else:
#             return_tuple = object_pose_pred
#         return return_tuple