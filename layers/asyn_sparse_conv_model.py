from typing import Dict, List, Tuple, Union

import torch.nn

import sparseconvnet as scn
from sparseconvnet.utils import *
from layers.conv_layer_2D import asynSparseConvolution2D, asynNonValidSparseConvolution2D
from layers.conv_layer_2D_cpp import asynSparseConvolution2Dcpp
from layers.max_pool import asynMaxPool
import torch.nn.functional as F
from dataloader.dataset import NCaltech101
if __name__ == "__main__":
    from training.trainer import AbstractTrainer

from layers.site_enum import Sites


class SparseConvolutionSamePadding(scn.Convolution):
    """
    (inefficiently) implements 'same' padding for sparse convolutions. More efficient implementation would require
    changes to the underlying C implementation, as the exposed python bindings don't provide the methods to manipulate
    sparse tensors sufficiently.
    """

    def __init__(self, batch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding_offset = torch.cat((self.filter_size // 2, torch.LongTensor([0])))
        # print(f"self.padding_offset: {self.padding_offset}")
        self.batch_size = batch_size

    def input_spatial_size(self, out_size):
        """
        computes the required spatial size of the input for a given output spatial size
        :param out_size: output spatial size
        :return: required input spatial size
        """
        return out_size * self.filter_stride - (self.filter_stride - 1)

    def forward(self, input):
        """
        applies 'same' padding by converting sparse input tensor to dense tensor, padding the dense tensor, then
        converting byck to sparse tensor. Then performs sparse convolution.
        :param input: sparse input Tensor
        :return: result of convolution on padded input
        """
        input_layer = scn.InputLayer(self.dimension, input.spatial_size + (self.filter_size // 2) * 2, mode=2)
        locations = input.metadata.getSpatialLocations(input.spatial_size)
        padded_locations = locations + self.padding_offset
        padded_input = input_layer([padded_locations, input.features, self.batch_size])
        result = super().forward(padded_input)
        return result


# SubmanifolConvolutions already apply same padding, as only active neighbour sites are considered, and therefore the
# always inactive rim around the tensor does not need to be known


def expand_cfg(
    cfg: Union[List[Union[Dict, List[Union[Dict, int]]]], List[Union[Union[Tuple, str], List[Union[Tuple, str, int]]]]]
):
    """
    expands repeated layers in the layer specification
    :param cfg: network configuration possibly containing repeated layers as List[[layer_config], [# repeats]]
    :return: network configuration containing exactly one tuple for each layer in the model
    """
    cfg_expanded = []
    for v in cfg:
        if isinstance(v, list):
            times = v[-1]
            for _ in range(times):
                cfg_expanded = cfg_expanded + v[:-1]
        else:
            cfg_expanded.append(v)
    return cfg_expanded


def convert_config_to_layer_list(cfg: List[Union[Tuple, List[Union[Tuple, int]]]], nr_input_channels: int):
    """
    Convert layer shorthands in configuration of the convolutional block to full independent specifications for each
    layer.
    :param cfg: expanded cfg, that is, no list as element
    :param nr_input_channels: number of channels of the input tensors
    :return: List of one full specifications for each layer noted in cfg
    """
    layer_list = []
    in_channels = nr_input_channels
    for v in cfg:
        expanded_item = {}
        if v == 'M':  # Max pool
            expanded_item['layer'] = 'SparseMaxPool'
            expanded_item['dimension'] = 2
            expanded_item['filter_size'] = 2
            expanded_item['filter_stride'] = 2
            expanded_item['padding_mode'] = 'valid'
            layer_list.append(expanded_item)
        elif isinstance(v, tuple):  # conv
            if len(v) == 3:
                # Conv (kernel_size, out_channels, stride)
                expanded_item['layer'] = 'SparseConv'
                expanded_item['dimension'] = 2
                expanded_item['filter_size'] = v[0]
                expanded_item['nOut'] = v[1]
                expanded_item['filter_stride'] = v[2]
                expanded_item['nIn'] = in_channels
                layer_list.append(expanded_item)
            else:
                # Conv (kernel_size, out_channels)
                expanded_item['layer'] = 'ValidSparseConv'
                expanded_item['dimension'] = 2
                expanded_item['filter_size'] = v[0]
                expanded_item['nOut'] = v[1]
                expanded_item['nIn'] = in_channels
                layer_list.append(expanded_item)
                layer_list.append({'layer': 'SparseBatchNorm', 'num_features': v[1], 'eps': 1e-4, 'momentum': 0.9})
            layer_list.append({'layer': 'LeakyRelu', 'negative_slope': 0.1})
            in_channels = v[1]
    return layer_list


class SparseConvModel(torch.nn.Module):
    """
    general synchronous sparse convolutional model
    """

    def __init__(
            self, config: Union[List[Union[Dict, List[Union[Dict, int]]]], List[Union[Tuple, List[Union[Tuple, int]]]]],
            dense_layers: List[Dict], nr_input_channels: int,
            batch_size: int, dense_input_channels: int,
            cnn_spatial_output_size: Tuple[int, int], use_bias: bool = True,
            device=torch.device('cpu')
    ):
        super(SparseConvModel, self).__init__()

        self.device = device
        self.use_bias = use_bias

        self.batch_size = batch_size
        self.cnn_spatial_output_size = cnn_spatial_output_size
        self.dense_input_channels = dense_input_channels
        self.dense_input_feature_size = (
            cnn_spatial_output_size[0] * cnn_spatial_output_size[1] * self.dense_input_channels
        )

        self.sparseModel = scn.Sequential()
        self.denseModel = torch.nn.Sequential()

        config = expand_cfg(config)
        if isinstance(config[0], str) or isinstance(config[0], tuple):
            assert nr_input_channels is not None, "parameter 'nr_input_channels' required when passing a list of " \
                                                  "tuples as config"
            self.sparse_layer_list = convert_config_to_layer_list(config, nr_input_channels)
        else:
            self.sparse_layer_list = config
        self.dense_layer_list = dense_layers

        self.make_layers()

        self.input_spatial_size = self.sparseModel.input_spatial_size(torch.LongTensor(self.cnn_spatial_output_size))
        print(f"self.spatial_size: {self.input_spatial_size}")
        self.inputLayer = scn.InputLayer(dimension=2, spatial_size=self.input_spatial_size, mode=2).to(self.device)

    def get_input_spacial_size(self):
        return torch.tensor(list(self.input_spatial_size))

    def forward(self, x):
        x = self.inputLayer(x)
        x = self.sparseModel(x)
        x = x.view(-1, self.dense_input_feature_size)
        x = self.denseModel(x)
        return x

    def convert_layer_def_to_kwargs(self, layer: dict, conv: bool):
        """
        convert a layer specification to the keyword arguments that will be passed to the initializer of the
        respective layer type
        :param layer: full layer specification generated from networg configuration
        :param conv: is a convolutional layer
        :return: kwargs to pass to __init__ of layer['layer']
        """
        kwargs = layer.copy()
        del kwargs['layer']
        if conv:
            kwargs['bias'] = self.use_bias
        return kwargs

    def make_layers(self):
        """
        Make layers based on configuration.
        :return: nn sequential module
        """
        for layer in self.sparse_layer_list:

            if layer['layer'] == 'ValidSparseConv':
                kwargs = self.convert_layer_def_to_kwargs(layer, True)
                self.sparseModel.add(
                    scn.SubmanifoldConvolution(
                        **kwargs
                    )
                )
                self.sparseModel[-1].to(self.device)

            elif layer['layer'] == 'SparseConv':
                kwargs = self.convert_layer_def_to_kwargs(layer, True)
                kwargs['batch_size'] = self.batch_size
                self.sparseModel.add(
                    SparseConvolutionSamePadding(
                        **kwargs
                    )
                )
                self.sparseModel[-1].to(self.device)

            elif layer['layer'] == 'SparseMaxPool':
                self.sparseModel.add(
                    scn.MaxPooling(
                        dimension=2,
                        pool_size=layer['filter_size'],
                        pool_stride=layer['filter_stride']
                    )
                )
                self.sparseModel[-1].to(self.device)

            elif layer['layer'] == 'SparseBatchNorm':
                kwargs = self.convert_layer_def_to_kwargs(layer, False)
                kwargs['nPlanes'] = kwargs['num_features']
                del kwargs['num_features']
                self.sparseModel.add(
                    scn.BatchNormalization(
                        **kwargs
                    )
                )
                self.sparseModel[-1].to(self.device)

            elif layer['layer'] == 'Relu':
                self.sparseModel.add(scn.ReLU())
                self.sparseModel[-1].to(self.device)

            elif layer['layer'] == 'LeakyRelu':
                self.sparseModel.add(
                    scn.LeakyReLU(
                        leak=layer['negative_slope']
                    )
                )
                self.sparseModel[-1].to(self.device)

            else:
                raise ValueError(f"Encountered unexpected sparse Layer {layer['layer']}")

        self.sparseModel.add(scn.SparseToDense(dimension=2, nPlanes=self.dense_input_channels))

        dense_layers = []
        for layer in self.dense_layer_list:

            if layer['layer'] == 'Conv':
                kwargs = self.convert_layer_def_to_kwargs(layer, True)
                dense_layers.append(
                    torch.nn.Conv2d(
                        **kwargs
                    )
                )

            elif layer['layer'] == 'BatchNorm':
                kwargs = self.convert_layer_def_to_kwargs(layer, False)
                dense_layers.append(
                    torch.nn.BatchNorm1d(
                        **kwargs
                    )
                )

            elif layer['layer'] == 'FC':
                kwargs = self.convert_layer_def_to_kwargs(layer, False)
                dense_layers.append(
                    torch.nn.Linear(
                        **kwargs
                    )
                )

            elif layer['layer'] == 'Relu':
                dense_layers.append(torch.nn.ReLU())

            elif layer['layer'] == 'LeakyRelu':
                kwargs = self.convert_layer_def_to_kwargs(layer, False)
                dense_layers.append(
                    torch.nn.LeakyReLU(
                        **kwargs
                    )
                )

            else:
                raise ValueError(f"Encountered unexpected dense Layer {layer['layer']}")

        self.denseModel = torch.nn.Sequential(*dense_layers)


class AsynSparseConvModel(torch.nn.Module):
    """
    general asynchronous sparse convolutional model
    """

    def __init__(
            self, config: Union[List[Union[Dict, List[Union[Dict, int]]]], List[Union[Tuple, List[Union[Tuple, int]]]]],
            dense_layers: List[Dict], nr_input_channels: int, use_bias: bool = True, device=torch.device('cpu'),
            cpp: bool = False
    ):
        super(AsynSparseConvModel, self).__init__()

        self.device = device
        self.use_bias = use_bias
        self.cpp = cpp

        self.asyn_layers = []
        self.dense_layers = []

        config = expand_cfg(config)
        if isinstance(config[0], str) or isinstance(config[0], tuple):
            assert nr_input_channels is not None, "parameter 'nr_input_channels' required when passing a list of " \
                                                  "tuples as config"
            self.asyn_layer_list = convert_config_to_layer_list(config, nr_input_channels)
        else:
            self.asyn_layer_list = config
        self.dense_layer_list = dense_layers
        print('layer list created.')

        # TODO option for no padding

        self.rule_book_start = [True]

        self.create_asyn_sparse_network()

    def forward_async(self, x_asyn):
        """
        asynchronously update only the convolutional block of the network if no prediction is required, but only the
        internal state should be updated for this sample-subsequence
        :param x_asyn: part of an asynchronous sample, consisting of event histogram and update locations
        :return: predicted tensor at the end of the convolutional block (for debugging purposes and use in
                 self.forward(), not a prediction)
        """

        # eliminate duplicate update locations (semantically irrelevent within one sequence, and causing wrong results
        # if present)
        x_asyn[0] = torch.unique(x_asyn[0], dim=0)

        x_asyn[0] = x_asyn[0].double()
        x_asyn[1] = x_asyn[1].double()

        for i, layer in enumerate(self.asyn_layers):
            # print(x_asyn[1].shape)
            # print(f"Layer {j}: {str(layer)}")
            # print(
            #     f"\tcomputing async layer {i}/{len(self.asyn_layers) - 1} "
            #     f"(total {i}/{len(self.asyn_layers) + len(self.dense_layers) - 1}): {str(layer)}"
            # )  # TODO debug
            layer_name = self.asyn_layer_list[i]['layer']
            if (
                layer_name == 'ValidSparseConv'
                and
                self.cpp
            ):
                if self.rule_book_start[i]:
                    x_asyn = layer.forward(update_location=x_asyn[0].to(self.device).detach().numpy(),
                                           feature_map=x_asyn[1].to(self.device).detach().numpy(),
                                           active_sites_map=None,
                                           rule_book=None)
                else:
                    x_asyn = layer.forward(update_location=x_asyn[0],
                                           feature_map=x_asyn[1].to(self.device).detach().numpy(),
                                           active_sites_map=x_asyn[2].numpy(),
                                           rule_book=x_asyn[3])
                x_asyn = list(x_asyn)
                x_asyn[2] = torch.from_numpy(x_asyn[2].astype(np.int32))
                x_asyn[1] = torch.from_numpy(x_asyn[1].astype(np.double))
            elif (
                layer_name == 'ValidSparseConv'
                or
                layer_name == 'SparseConv'
            ):
                if self.rule_book_start[i]:
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

            elif layer_name == 'SparseMaxPool':
                # Get all the updated/changed locations
                if x_asyn[2] is not None:
                    changed_locations = (x_asyn[2] > Sites.ACTIVE_SITE.value).nonzero()
                    x_asyn = layer.forward(update_location=changed_locations.long(), feature_map=x_asyn[1])
                else:
                    if 0 in x_asyn[0].shape:
                        x_asyn = layer.forward(update_location=torch.empty((0, 2)).long(), feature_map=x_asyn[1])
                    else:
                        x_asyn = layer.forward(update_location=x_asyn[0].long(), feature_map=x_asyn[1])

            elif layer_name == 'SparseBatchNorm':
                self.apply_asyn_batch_norm(layer, x_asyn[2].clone(), x_asyn[1])

            elif layer_name == 'Relu':
                x_asyn[1] = layer(x_asyn[1])  # simply applies layer to features

            elif layer_name == 'LeakyRelu':
                x_asyn[1] = layer(x_asyn[1])  # simply applies layer to features

            else:
                raise ValueError(f"invalid layer: '{layer_name}'")

        return x_asyn

    def forward(self, x_asyn):
        """
        compute full forward pass for this sample sub-sequence, also processing the fully-connected tail of the model
        and producing a prediction
        :param x_asyn: art of an asynchronous sample, consisting of event histogram and update locations
        :return: prediction of the model
        """

        # update and compute convolutional block of the model
        x_asyn = self.forward_async(x_asyn)

        # compute fully-connected tail of the model
        for i, layer in enumerate(self.dense_layers):
            # print(x_asyn[1].shape)
            # print(f"Layer {j}: {str(layer)}")
            # print(
            #     f"\tcomputing sync layer {i}/{len(self.dense_layers) - 1} "
            #     f"(total {i}/{len(self.asyn_layers) + len(self.dense_layers) - 1}): {str(layer)}"
            # )  # TODO debug
            layer_name = self.dense_layer_list[i]['layer']

            if layer_name == 'ClassicC':
                conv_input = x_asyn[1].unsqueeze(0).permute(0, 3, 1, 2)
                conv_output = layer(conv_input)
                x_asyn = [None] * 2
                x_asyn[1] = conv_output.squeeze(0).permute(1, 2, 0)

            elif layer_name == 'BatchNorm':
                x_asyn[1] = layer(x_asyn[1])  # simply applies layer to features

            elif layer_name == 'FC':
                if x_asyn[1].ndim == 3:
                    fc_output = layer(x_asyn[1].permute(2, 0, 1).flatten().unsqueeze(0))
                else:
                    fc_output = layer(x_asyn[1].unsqueeze(0))
                x_asyn = [None] * 5
                x_asyn[1] = fc_output.squeeze(0)

            elif layer_name == 'Relu':
                x_asyn[1] = layer(x_asyn[1])  # simply applies layer to features

            elif layer_name == 'LeakyRelu':
                x_asyn[1] = layer(x_asyn[1])  # simply applies layer to features

            else:
                raise ValueError(f"invalid layer: '{layer_name}'")

        return x_asyn

    def reset(self):
        """
        reset internal state of the model (of the async convolutional layers) to clean state, necessary between
        predictions of independent samples
        :return: None
        """
        for i, layer in enumerate(self.asyn_layers):
            layer_name = self.asyn_layer_list[i]['layer']
            if (
                layer_name == 'SparseConv'
                or (layer_name == 'ValidSparseConv' and not self.cpp)
                or layer_name == 'SparseMaxPool'
            ):
                layer.reset()

    def convert_layer_def_to_kwargs(
            self, layer: dict, conv: bool, current_asyn_layer: str = None, last_asyn_layer: str = None
    ):
        """
        convert a layer specification to the keyword arguments that will be passed to the initializer of the
        respective layer type
        :param layer: full layer specification generated from networg configuration
        :param conv: is a convolutional layer
        :param current_asyn_layer: type of the current async convolutional layer
        :param last_asyn_layer: type of the last async convolutional layer
        :return: kwargs to pass to __init__ of layer['layer']
        """
        kwargs = layer.copy()
        del kwargs['layer']
        if conv:
            kwargs['device'] = self.device
            first_layer = last_asyn_layer != current_asyn_layer or self.rule_book_start[-1]
            kwargs['first_layer'] = first_layer
            kwargs['use_bias'] = self.use_bias
        return kwargs

    def create_asyn_sparse_network(self):
        """
        Make layers based on configuration.
        :return: nn sequential module
        """
        # first make asynchronous convolutional block
        last_asyn_layer = ""
        for layer in self.asyn_layer_list:

            if layer['layer'] == 'ValidSparseConv':
                current_asyn_layer = f"VSC_{layer['filter_size']}"
                kwargs = self.convert_layer_def_to_kwargs(layer, True, current_asyn_layer, last_asyn_layer)
                if self.cpp:
                    del kwargs['device']
                    self.asyn_layers.append(
                        asynSparseConvolution2Dcpp(
                            **kwargs
                        )
                    )
                else:
                    self.asyn_layers.append(
                        asynSparseConvolution2D(
                            **kwargs
                        )
                    )
                self.rule_book_start.append(False)
                last_asyn_layer = current_asyn_layer

            elif layer['layer'] == 'SparseConv':
                current_asyn_layer = f"SC_{layer['filter_size']}"
                kwargs = self.convert_layer_def_to_kwargs(layer, True, current_asyn_layer, last_asyn_layer)
                self.asyn_layers.append(
                    asynNonValidSparseConvolution2D(
                        **kwargs
                    )
                )
                self.rule_book_start.append(False)
                last_asyn_layer = current_asyn_layer

            elif layer['layer'] == 'SparseMaxPool':
                kwargs = self.convert_layer_def_to_kwargs(layer, False)
                kwargs['device'] = self.device
                self.asyn_layers.append(
                    asynMaxPool(
                        **kwargs
                    )
                )
                self.rule_book_start.append(True)

            elif layer['layer'] == 'SparseBatchNorm':
                kwargs = self.convert_layer_def_to_kwargs(layer, False)
                self.asyn_layers.append(
                    torch.nn.BatchNorm1d(
                        **kwargs
                    )
                )
                self.asyn_layers[-1].double()
                self.asyn_layers[-1].to(self.device)
                # TODO relu params
                self.rule_book_start.append(self.rule_book_start[-1])

            elif layer['layer'] == 'Relu':
                self.asyn_layers.append(torch.nn.ReLU())
                self.asyn_layers[-1].to(self.device)
                self.rule_book_start.append(self.rule_book_start[-1])

            elif layer['layer'] == 'LeakyRelu':
                kwargs = self.convert_layer_def_to_kwargs(layer, False)
                self.asyn_layers.append(
                    torch.nn.LeakyReLU(
                        **kwargs
                    )
                )
                self.asyn_layers[-1].to(self.device)
                self.rule_book_start.append(self.rule_book_start[-1])

            else:
                raise ValueError(f"Encountered unexpected asynchronous Layer {layer['layer']}")

        # then create fully connected tail
        for layer in self.dense_layer_list:

            if layer['layer'] == 'Conv':
                kwargs = self.convert_layer_def_to_kwargs(layer, False)
                self.dense_layers.append(
                    torch.nn.Conv2d(
                        bias=self.use_bias,
                        **kwargs
                    )
                )
                self.dense_layers[-1].to(self.device)

            elif layer['layer'] == 'BatchNorm':
                kwargs = self.convert_layer_def_to_kwargs(layer, False)
                self.dense_layers.append(
                    torch.nn.BatchNorm1d(
                        **kwargs
                    )
                )
                self.dense_layers[-1].double()
                self.dense_layers[-1].to(self.device)
                # TODO relu params

            elif layer['layer'] == 'FC':
                kwargs = self.convert_layer_def_to_kwargs(layer, False)
                self.dense_layers.append(
                    torch.nn.Linear(
                        **kwargs
                    ).double()
                )
                self.dense_layers[-1].to(self.device)

            elif layer['layer'] == 'Relu':
                self.dense_layers.append(torch.nn.ReLU())

            elif layer['layer'] == 'LeakyRelu':
                kwargs = self.convert_layer_def_to_kwargs(layer, False)
                self.dense_layers.append(
                    torch.nn.LeakyReLU(
                        **kwargs
                    )
                )
                self.dense_layers[-1].to(self.device)

            else:
                raise ValueError(f"Encountered unexpected dense Layer {layer['layer']}")

    def set_weights_equal(self, fb_model: SparseConvModel):
        """
        Sets weights and biases equal to those of an identically configured (synchronous and thus trainable) sparse
        convolutional model
        :param fb_model: equivalent synchronous sparse convolutional model
        :return: None
        """
        # process asyn layers
        for i, layer in enumerate(self.asyn_layers):
            # print(f"matching asyn Layer {asyn_layer_i} with syn Layer {syn_layer_i}.")
            # print(f"asyn Layer: \t{str(self.asyn_layers[asyn_layer_i])}")
            layer_name = self.asyn_layer_list[i]['layer']
            if layer_name == 'ValidSparseConv':
                # conv
                # print(f"syn Layer: \t{str(fb_model.sparseModel[syn_layer_i])}")
                layer.weight.data = fb_model.sparseModel[i].weight.squeeze(1).to(self.device)
            elif layer_name == 'SparseConv':
                # conv
                # print(f"syn Layer: \t{str(fb_model.sparseModel[syn_layer_i])}")
                layer.weight.data = fb_model.sparseModel[i].weight.squeeze(1).to(self.device)
            elif layer_name == 'SparseMaxPool':
                # print(f"syn Layer: \t{str(fb_model.sparseModel[syn_layer_i])}")
                pass
            elif layer_name == 'Relu':
                # skip 'ClassicRelu'
                pass
            elif layer_name == 'LeakyRelu':
                # skip 'ClassicRelu'
                pass
            elif layer_name == 'SparseBatchNorm':
                # BNRelu
                layer.weight.data = fb_model.sparseModel[i].weight.data.double().to(self.device)
                layer.bias.data = fb_model.sparseModel[i].bias.data.double().to(self.device)
                layer.running_mean.data = fb_model.sparseModel[i].running_mean.data.double().to(self.device)
                layer.running_var.data = fb_model.sparseModel[i].running_var.data.double().to(self.device)
                layer.eval()
                fb_model.sparseModel[i].eval()
            else:
                raise ValueError(f"Encountered unexpected Layer {layer_name}")
        # process dense layers
        for i, layer in enumerate(self.dense_layers):
            # print(f"matching asyn Layer {asyn_layer_i} with syn Layer {syn_layer_i}.")
            # print(f"asyn Layer: \t{str(self.asyn_layers[asyn_layer_i])}")
            layer_name = self.dense_layer_list[i]['layer']
            if layer_name == 'ClassicC':
                # conv
                layer.weight.data = fb_model.denseModel[i].weight.squeeze(1).to(self.device)
                # print(f"syn Layer: \t{str(fb_model.sparseModel[syn_layer_i])}")
                # weight_fb = fb_model.sparseModel[syn_layer_i].weight.squeeze(1).to(self.device).permute(2, 1, 0)
                # kernel_size = int(np.sqrt(weight_fb.shape[2]))
                # self.asyn_layers[asyn_layer_i].weight.data = weight_fb.reshape(
                # [weight_fb.shape[0],
                #  weight_fb.shape[1], kernel_size, kernel_size]).double()
            elif layer_name == 'FC':
                layer.weight.data = fb_model.denseModel[i].weight.data.double().to(self.device)
                layer.bias.data = fb_model.denseModel[i].bias.data.double().to(self.device)
            elif layer_name == 'MaxPool':
                # print(f"syn Layer: \t{str(fb_model.sparseModel[syn_layer_i])}")
                pass
            elif layer_name == 'Relu':
                # skip 'ClassicRelu'
                pass
            elif layer_name == 'LeakyRelu':
                # skip 'ClassicRelu'
                pass
            elif layer_name == 'BatchNorm':
                # BNRelu
                layer.weight.data = fb_model.denseModel[i].weight.data.double().to(self.device)
                layer.bias.data = fb_model.denseModel[i].bias.data.double().to(self.device)
                layer.running_mean.data = fb_model.denseModel[i].running_mean.data.double().to(self.device)
                layer.running_var.data = fb_model.denseModel[i].running_var.data.double().to(self.device)
                layer.eval()
                fb_model.denseModel[i].eval()
            else:
                raise ValueError(f"Encountered unexpected Layer {layer_name}")

    @staticmethod
    def generate_asyn_input(new_batch_events, spatial_dimensions, original_shape):
        """Generates the asynchronous input for the sparse VGG, which is consistent with training input"""
        list_spatial_dimensions = [spatial_dimensions.cpu().numpy()[0], spatial_dimensions.cpu().numpy()[1]]
        new_histogram = NCaltech101.generate_event_histogram(new_batch_events, original_shape)
        new_histogram = torch.from_numpy(new_histogram[np.newaxis, :, :])

        new_histogram = torch.nn.functional.interpolate(new_histogram.permute(0, 3, 1, 2), list_spatial_dimensions)
        new_histogram = new_histogram.permute(0, 2, 3, 1)

        update_locations, features = AbstractTrainer.denseToSparse(new_histogram)

        return update_locations, new_histogram.squeeze(0)

    @staticmethod
    def apply_asyn_batch_norm(layer, bn_active_sites, feature_map_to_update):
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

    def print_layers(self):
        """
        print the architecture of the asynchronous convolutional block
        :return:
        """
        for j, layer in enumerate(self.asyn_layers):
            print(str(layer))
