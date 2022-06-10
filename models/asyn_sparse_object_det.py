import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F

from dataloader.dataset import NCaltech101
from training.trainer import AbstractTrainer
from layers.site_enum import Sites
from layers.conv_layer_2D import asynSparseConvolution2D
from layers.max_pool import asynMaxPool


class asynSparseObjectDet:
    def __init__(self, nr_classes=101, nr_box=2, input_channels=2, small_out_map=True, layer_list=None, device=torch.device('cpu')):
        self.device = device
        self.use_bias = False
        self.asyn_layers = []
        self.nr_box = nr_box
        self.nr_classes = nr_classes
        if layer_list is not None:
            self.layer_list = layer_list
        else:
            sparse_out_channels = 256

            if small_out_map:
                self.cnn_spatial_output_size = [5, 7]
            else:
                self.cnn_spatial_output_size = [6, 8]
            spatial_size_product = self.cnn_spatial_output_size[0] * self.cnn_spatial_output_size[1]
            # layer definition analogously to 'facebook_sparse_object_det.py'
            self.layer_list = [['C', input_channels,    16], ['BNRelu'], ['C', 16, 16],   ['BNRelu'], ['MP'],
                               ['C', 16,   32], ['BNRelu'], ['C', 32, 32],   ['BNRelu'], ['MP'],
                               ['C', 32,   64], ['BNRelu'], ['C', 64, 64],   ['BNRelu'], ['MP'],
                               ['C', 64,  128], ['BNRelu'], ['C', 128, 128], ['BNRelu'], ['MP'],
                               ['C', 128, 256], ['BNRelu'], ['C', 256, 256], ['BNRelu'],
                               ['ClassicC', 256, sparse_out_channels, 3, 2],
                               ['ClassicBNRelu'],
                               ['ClassicFC', sparse_out_channels*3*2, 1024],
                               ['ClassicRelu'],
                               ['ClassicFC', 1024, spatial_size_product*(nr_classes + 5*self.nr_box)],
                               ]
        # self.layer_list[0][1] = input_channels  # set explixitly in layer def
        # self.layer_list[-1][2] = nr_classes  # set explixitly in layer def

        self.rule_book_start = [True]

        self.createAsynVGG()

    def forward(self, x_asyn):
        """Apply asynchronous layers"""
        for j, layer in enumerate(self.asyn_layers):
            # print('Layer Name: %s' % layer_list[j][0])
            if self.layer_list[j][0] == 'C':
                if self.rule_book_start[j]:
                    x_asyn = layer.forward(update_location=x_asyn[0].to(self.device),
                                           feature_map=x_asyn[1].to(self.device),
                                           active_sites_map=None,
                                           rule_book_input=None,
                                           rule_book_output=None)
                else:
                    x_asyn = layer.forward(update_location=x_asyn[0].to(self.device),
                                           feature_map=x_asyn[1].to(self.device),
                                           active_sites_map=x_asyn[2],
                                           rule_book_input=x_asyn[3],
                                           rule_book_output=x_asyn[4])
                x_asyn = list(x_asyn)

            elif self.layer_list[j][0] == 'ClassicC':
                conv_input = x_asyn[1].unsqueeze(0).permute(0, 3, 1, 2)
                conv_output = layer(conv_input)
                x_asyn = [None] * 2
                x_asyn[1] = conv_output.squeeze(0).permute(1, 2, 0)
            elif self.layer_list[j][0] == 'MP':
                # Get all the updated/changed locations
                changed_locations = (x_asyn[2] > Sites.ACTIVE_SITE.value).nonzero()
                x_asyn = layer.forward(update_location=changed_locations.long(), feature_map=x_asyn[1])

            elif self.layer_list[j][0] == 'BNRelu':
                self.applyBatchNorm(layer, x_asyn[2].clone(), x_asyn[1])
                x_asyn[1] = F.relu(x_asyn[1])
            elif self.layer_list[j][0] == 'ClassicBNRelu':
                bn_input = x_asyn[1].permute(2, 0, 1).unsqueeze(0)
                bn_output = F.relu(layer(bn_input))
                x_asyn = [None] * 2
                x_asyn[1] = bn_output.squeeze(0).permute(1, 2, 0)
            elif self.layer_list[j][0] == 'ClassicFC':
                if x_asyn[1].ndim == 3:
                    fc_output = layer(x_asyn[1].permute(2, 0, 1).flatten().unsqueeze(0))
                else:
                    fc_output = layer(x_asyn[1].unsqueeze(0))
                x_asyn = [None] * 5
                x_asyn[1] = fc_output.squeeze(0)
            elif self.layer_list[j][0] == 'ClassicRelu':
                x_asyn[1] = layer(x_asyn[1])  # TODO ??

        x_asyn[1] = x_asyn[1].view(
            (
                self.cnn_spatial_output_size[0],
                self.cnn_spatial_output_size[1],
                (self.nr_classes + 5*self.nr_box)
            )
        ).permute(1, 2, 0)

        return x_asyn

    def createAsynVGG(self):
        """Creates a asynchronous VGG"""
        for j, i_layer in enumerate(self.layer_list):
            if i_layer[0] == 'C':
                self.asyn_layers.append(asynSparseConvolution2D(dimension=2, nIn=i_layer[1], nOut=i_layer[2],
                                                                filter_size=3, first_layer=self.rule_book_start[-1],
                                                                use_bias=self.use_bias, device=self.device))
                self.rule_book_start.append(False)
            elif i_layer[0] == 'ClassicC':
                self.asyn_layers.append(torch.nn.Conv2d(in_channels=i_layer[1], out_channels=i_layer[2],
                                                        kernel_size=i_layer[3], stride=i_layer[4], bias=False))
                self.rule_book_start.append(True)
            elif i_layer[0] == 'BNRelu':
                self.asyn_layers.append(torch.nn.BatchNorm1d(self.layer_list[j-1][2], eps=1e-4, momentum=0.9))
                self.rule_book_start.append(self.rule_book_start[-1])
            elif i_layer[0] == 'ClassicBNRelu':
                self.asyn_layers.append(torch.nn.BatchNorm2d(self.layer_list[j-1][2], eps=1e-4, momentum=0.9))
                self.rule_book_start.append(self.rule_book_start[-1])
            elif i_layer[0] == 'MP':
                self.asyn_layers.append(asynMaxPool(dimension=2, filter_size=3, filter_stride=2, padding_mode='valid',
                                                    device=self.device))
                self.rule_book_start.append(True)

            elif i_layer[0] == 'ClassicFC':
                self.asyn_layers.append(torch.nn.Linear(in_features=i_layer[1], out_features=i_layer[2]))
                self.rule_book_start.append(True)

            elif i_layer[0] == 'ClassicRelu':
                self.asyn_layers.append(torch.nn.ReLU())
                self.rule_book_start.append(True)

    def setWeightsEqual(self, fb_model):
        """Sets the different weights and biases equal"""
        dense_layer_counter = 0
        for j, i_layer in enumerate(self.layer_list):
            layer_name = i_layer[0]
            if layer_name == 'C':
                self.asyn_layers[j].weight.data = fb_model.sparseModel[j].weight.squeeze(1).to(self.device)
            if layer_name == 'ClassicC':
                weight_fb = fb_model.sparseModel[j].weight.squeeze(1).to(self.device).permute(2, 1, 0)
                self.asyn_layers[j].weight.data = weight_fb.reshape([weight_fb.shape[0],
                                                                     weight_fb.shape[1], 3, 3]).double()
            if layer_name == 'BNRelu' or layer_name == 'ClassicBNRelu':
                self.asyn_layers[j].weight.data = fb_model.sparseModel[j].weight.data.double().to(self.device)
                self.asyn_layers[j].bias.data = fb_model.sparseModel[j].bias.data.double().to(self.device)
                self.asyn_layers[j].running_mean.data = fb_model.sparseModel[j].running_mean.data.double().to(self.device)
                self.asyn_layers[j].running_var.data = fb_model.sparseModel[j].running_var.data.double().to(self.device)
                self.asyn_layers[j].eval()
                fb_model.sparseModel[j].eval()
            if layer_name == 'ClassicFC':
                if dense_layer_counter == 0:
                    fb_layer = fb_model.linear_1
                else:
                    fb_layer = fb_model.linear_2
                dense_layer_counter += 1
                self.asyn_layers[j].weight.data = fb_layer.weight.data.double().to(self.device)
                self.asyn_layers[j].bias.data = fb_layer.bias.data.double().to(self.device)
            if layer_name == 'ClassicRelu':
                pass

    def generateAsynInput(self, new_batch_events, spatial_dimensions, original_shape):
        """Generates the asynchronous input for the sparse VGG, which is consistent with training input"""
        list_spatial_dimensions = [spatial_dimensions.cpu().numpy()[0], spatial_dimensions.cpu().numpy()[1]]
        new_histogram = NCaltech101.generate_event_histogram(new_batch_events, original_shape)
        new_histogram = torch.from_numpy(new_histogram[np.newaxis, :, :])

        new_histogram = torch.nn.functional.interpolate(new_histogram.permute(0, 3, 1, 2), list_spatial_dimensions)
        new_histogram = new_histogram.permute(0, 2, 3, 1)

        update_locations, features = AbstractTrainer.denseToSparse(new_histogram)

        return update_locations, new_histogram.squeeze(0)

    def applyBatchNorm(self, layer, bn_active_sites, feature_map_to_update):
        """
        Applies the batch norm layer to the the sparse features.

        :param layer: torch.nn.BatchNorm1d layer
        :param bn_active_sites: location of the active sites
        :param feature_map_to_update: feature map, result is stored in place to the tensor
        """
        bn_active_sites[bn_active_sites == Sites.NEW_INACTIVE_SITE.value] = 0
        active_sites = torch.squeeze(bn_active_sites.nonzero(), dim=-1)

        bn_input = torch.squeeze(feature_map_to_update[active_sites.split(1, dim=-1)], dim=1).T[None, :, :].double()
        sparse_bn_features = layer(bn_input)

        sparse_bn_features = torch.unsqueeze(torch.squeeze(sparse_bn_features, dim=0).T, dim=1)
        feature_map_to_update[active_sites.split(1, dim=-1)] = sparse_bn_features
