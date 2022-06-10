import torch
from abc import ABC, abstractmethod
import gc

from layers.site_enum import Sites


DEBUG = False


def print_debug(msg: str, level: int = 0):
    if DEBUG:
        print('\t' * (level + 2) + msg)


class asynSparseConvolution2DBase(ABC):
    def __init__(self, dimension, nIn, nOut, filter_size, stride=1, first_layer=False, use_bias=False,
                 device=torch.device('cpu')):
        """
        Constructs a convolution layer.

        :param dimension: spatial dimension of the convolution e.g. dimension=2 leads to kxk kernel
        :param nIn: number of channels in the input features
        :param nOut: number of output channels
        :param filter_size: kernel size
        :param first_layer: bool indicating if it is the first layer. Used for computing new inactive sites.
        """
        self.dimension = dimension
        self.nIn = nIn
        self.nOut = nOut
        self.filter_size = filter_size
        self.device = device

        self.filter_size_tensor = torch.LongTensor(dimension).fill_(self.filter_size).to(device)
        self.filter_volume = self.filter_size_tensor.prod().item()
        # weight initialization with standard distribution
        std = (2.0 / nIn / self.filter_volume)**0.5
        self.weight = torch.nn.Parameter(torch.Tensor(self.filter_volume, nIn, nOut).normal_(0, std)).to(device)
        #
        self.first_layer = first_layer
        self.use_bias = use_bias
        if use_bias:
            self.bias = torch.nn.Parameter(torch.Tensor(nOut).normal_(0, std)).to(device)
        # padding for active_sites_map lookups
        self.padding = [filter_size // 2] * dimension * 2
        # Construct lookup table for 1d kernel position to position in nd
        kernel_indices = torch.stack(torch.meshgrid([torch.arange(filter_size) for _ in range(dimension)]), dim=-1)
        self.kernel_indices = kernel_indices.reshape([self.filter_volume, dimension]).to(device)
        # define per-sample state accumulators
        self.output_feature_map = None
        self.old_input_feature_map = None

    def reset(self):
        """
        resets the per-sample state in preparation of processing a new sample
        :return: nothing
        """
        del self.output_feature_map
        del self.old_input_feature_map
        gc.collect()  # assist with cleanup
        # reset per-sample state accumulators
        self.output_feature_map = None
        self.old_input_feature_map = None

    @staticmethod
    def get_spatial_dimension_in(feature_map):
        return list(feature_map.shape[:-1])

    @abstractmethod
    def get_spatial_dimension_out(self, feature_map):
        pass

    @abstractmethod
    def requires_new_rulebooks(self):
        pass

    @staticmethod
    def reconstruct_update_locations(old_active_sites_map: torch.Tensor):
        """
        reconstruct list of update locations / events from the old active site map of the previous layer
        list of update locations is then required to build new rulebooks
        :param old_active_sites_map: active sites map returned by teh previous layer
        :return: list of reconstructed update events
        """
        locations = (
            (old_active_sites_map == Sites.ACTIVE_SITE.value)
            |
            (old_active_sites_map == Sites.UPDATED_SITE.value)
            |
            (old_active_sites_map == Sites.NEW_ACTIVE_SITE.value)
            |
            (old_active_sites_map == Sites.NEW_INACTIVE_SITE.value)
        ).nonzero()
        return locations

    @abstractmethod
    def filter_zero_update_locations(self, update_locations, feature_map):
        pass

    @abstractmethod
    def set_new_inactive_sites(self, active_sites_map, output_feature_map):
        pass

    @abstractmethod
    def filter_new_status_sites(self, i_active_sites, output_active_sites):
        pass

    def get_output_location(self, update_locations, index):
        """
        retrieve the update location with the specific index
        :param update_locations: locations of all newly active sites (=update_location[bool_new_active_site])
        :param index: index of the location of the current newly active site in the potentially filtered
            update_locations
        :return: location of the current newly active site
        """
        return update_locations[index, :]

    def compute_active_sites_to_update(self, padded_active_sites_map, kernel_update_location, reverse=False):
        """

        :param padded_active_sites_map: [spatial_size[0] + filter_size // 2, spatial_size[1] + filter_size // 2] padded
            site map indicating active sites
        :param kernel_update_location: [no_update_locations, filter_size**2, 2]: for each update location, the
            {filter_size**2} surrounding sites
        :param reverse:
            False:  compute output sites for given input sites
            True:   compute input sites for given output sites
        :return: [x, 2]
            first value: index of update location in update_locations
            second value: relative location (1d offset) of an active site inside the kernel around the update location
            may contain multiple active site values for one update location index
        """
        active_sites_to_update = torch.squeeze(padded_active_sites_map[kernel_update_location.split(1, dim=-1)], dim=-1)
        active_sites_to_update = active_sites_to_update.nonzero()
        return active_sites_to_update

    def filter_output_sites(self, padded_active_sites_map):
        pass

    @abstractmethod
    def mark_updated_sites(self, active_sites_map, update_location_indices):
        pass

    @abstractmethod
    def update_condition(self, output_active_site):
        pass

    @abstractmethod
    def filter_newly_active_sites(self, active_sites_map, input_indices, output_indices):
        pass

    @abstractmethod
    def need_to_reverse_newly_active_sites(self):
        pass

    def convert_output_locations(self, output_locations):
        """
        do nothing
        :param output_locations:
        :return: unchanged output_locations
        """
        return output_locations

    def forward(self, update_location, feature_map, active_sites_map=None, rule_book_input=None, rule_book_output=None):
        """
        Computes the a asynchronous sparse convolution layer based on the update_location

        :param update_location: tensor with shape [N_active, dimension]
        :param feature_map: tensor with shape [N, nIn]
        :param active_sites_map: tensor with shape [N]. Includes 1 - active side and 2 - location stored in Rulebooks,
                                  3 - new active site in input. 4 - new inactive site
        :param rule_book_input: list containing #(kernel_size) lists with locations in the input
        :param rule_book_output: list containing #(kernel_size) lists with locations in the output
        :return:
        """

        # TODO
        # self.output_feature_map = torch.Tensor([[[-2.9570],
        #          [ 7.9399]],
        #
        #         [[ 1.2877],
        #          [ 5.1855]]])
        # update_location = torch.Tensor([[1, 2],
        #         [0, 2],
        #         [3, 3],
        #         [0, 0],
        #         [2, 0],
        #         [1, 1]])
        # self.weight.data = torch.Tensor([
        #     [[0.4195]],
        #
        #     [[0.5676]],
        #
        #     [[-0.7081]],
        #
        #     [[0.6959]],
        #
        #     [[-0.0023]],
        #
        #     [[-0.0399]],
        #
        #     [[-0.0440]],
        #
        #     [[-0.3615]],
        #
        #     [[-0.2155]]
        # ])
        # self.old_input_feature_map = torch.Tensor([[[0.],
        #          [4.],
        #          [6.],
        #          [0.]],
        #
        #         [[0.],
        #          [0.],
        #          [1.],
        #          [0.]],
        #
        #         [[0.],
        #          [0.],
        #          [0.],
        #          [2.]],
        #
        #         [[0.],
        #          [3.],
        #          [0.],
        #          [0.]]])
        # feature_map = torch.Tensor([[[ 8.],
        #          [ 4.],
        #          [ 7.],
        #          [ 0.]],
        #
        #         [[ 0.],
        #          [ 1.],
        #          [-5.],
        #          [ 0.]],
        #
        #         [[ 1.],
        #          [ 0.],
        #          [ 0.],
        #          [ 2.]],
        #
        #         [[ 0.],
        #          [ 3.],
        #          [ 0.],
        #          [-8.]]])

        if active_sites_map is not None and (self.requires_new_rulebooks() or rule_book_input is None):
        # if not self.first_layer:
            # assert(update_location.nelement() == 0)
            rule_book_input = None
            rule_book_output = None
            update_location = self.reconstruct_update_locations(active_sites_map)
            active_sites_map = None

        update_location = update_location.long()
#        update_location = self.filter_zero_update_locations(update_location, feature_map)
        self.checkInputArguments(update_location, feature_map, active_sites_map)
        spatial_dimension_in = self.get_spatial_dimension_in(feature_map)
        spatial_dimension_out = self.get_spatial_dimension_out(feature_map)
        update_location_indices = update_location.split(1, dim=-1)

        if rule_book_input is None or len(rule_book_input) < self.filter_volume:
            rule_book_input = [[] for _ in range(self.filter_volume)]
            rule_book_output = [[] for _ in range(self.filter_volume)]
        if self.output_feature_map is None:
            self.output_feature_map = torch.zeros(spatial_dimension_out + [self.nOut]).to(self.device)
        if self.old_input_feature_map is None:
            self.old_input_feature_map = torch.zeros(spatial_dimension_in + [self.nIn]).to(self.device)
        if active_sites_map is None:
            # TODO ||feature_vec|| == 0 faster than not_zero(feature_vec)?
            active_sites_map = (torch.sum(feature_map**2, dim=-1) != 0).float().to(self.device)
            # active_sites_map = torch.squeeze((torch.sum(feature_map**2, dim=-1) != 0).float()).to(self.device)
            # Catch case if input feature is reduced to zero
            active_sites_map[update_location_indices] = Sites.ACTIVE_SITE.value

        if self.first_layer:
            # Set new deactivate sites to Sites.NEW_INACTIVE_SITE
            zero_input_update = torch.squeeze(torch.sum(feature_map[update_location_indices] ** 2, dim=-1) == 0, dim=-1)
            bool_new_active_site = (self.old_input_feature_map[update_location_indices] ** 2).sum(-1).squeeze(-1) == 0
        else:
            # zero_input_update = torch.squeeze(torch.sum(feature_map[update_location_indices] ** 2, dim=-1) == 0, dim=-1)  # TODO
            # bool_new_active_site = (self.old_input_feature_map[update_location_indices] ** 2).sum(-1).squeeze(-1) == 0
            zero_input_update = None
            bool_new_active_site = torch.zeros([update_location.shape[0]]).bool()

        print_debug("init complete.")

        """
        update rulebooks to determine which tensor locations have to be recomputed
        """

        if update_location.nelement() != 0:
            out = self.updateRuleBooks(active_sites_map, update_location, bool_new_active_site, zero_input_update,
                                       rule_book_input, rule_book_output, update_location_indices)
            rule_book_input, rule_book_output, new_update_events, active_sites_map = out
            # New update sites for next layer
            if len(new_update_events) == 0:
                new_update_events = torch.empty([0, self.dimension])
            else:
                new_update_events = torch.stack(new_update_events, dim=0)

        else:
            new_update_events = torch.empty([0, self.dimension])

        print_debug("rulebooks updated.")

        """
        compute convolution where necessary
        """

        # Compute update step with the rule book
        # Change to float64 for numerical stability
        feature_map = feature_map.double()
        old_input_feature_map = self.old_input_feature_map.double()
        output_feature_map = self.output_feature_map.double()

        # Create vector for ravel the output 2D indices to 1D flattened indices
        # Only valid for 2D
        # only for output site coordinates
        # extract lower spatial dimension of the feature map and extend it by new dimension with value 1
        # -> converts 2d coordinates into 1d revelled ones:
        # SUM((1, 1) * (512, 1)) = 513
        # SUM((1, 2) * (512, 1)) = 514
        # SUM((2, 1) * (512, 1)) = 1025
        flattened_indices_dim = torch.tensor(output_feature_map.shape[:-1], device=self.device)
        flattened_indices_dim = torch.cat((flattened_indices_dim[1:],
                                           torch.ones([1], dtype=torch.long, device=self.device)))

        print_debug("iterating kernels:")

        # process each kernel index once (rules are indexed by kernel index)
        for i_kernel in range(self.filter_volume):
            print_debug(f"computing kernel {i_kernel + 1}/{self.filter_volume}.", 1)
            # if no rules for this kernel index, skip
            if len(rule_book_input[i_kernel]) == 0:
                continue

            # extract site locations (2d indices) from rulebook(s)
            input_indices = torch.stack(rule_book_input[i_kernel], dim=0).long().split(1, dim=-1)
            output_indices = torch.stack(rule_book_output[i_kernel], dim=0).long().split(1, dim=-1)

            # create filter for sites that are not newly active
            bool_not_new_sites = self.filter_newly_active_sites(active_sites_map, input_indices, output_indices)

            # allow subclasses to modify the output locations
            output_indices = (
                self.convert_output_locations(output_indices[0]),
                self.convert_output_locations(output_indices[1])
            )

            # TODO why first squeeze, then just re-add teh removed dimension?
            # delta feature is full site value if site is newly active,
            # else value difference to last processed value at that site
            delta_feature = torch.squeeze(feature_map[input_indices], 1) - \
                            torch.squeeze(old_input_feature_map[input_indices], 1) * bool_not_new_sites

            # compute delta_feature times the weight of the current kernel index (for all input/output channels)
            update_term = torch.matmul(delta_feature[:, None, :],
                                       self.weight[None, i_kernel, :, :].double()).squeeze(dim=1)

            # flatten/ravel output site locations for use with torch.Tensor.index_add_(...)
            flattend_indices = torch.cat(output_indices, dim=-1)
            flattend_indices = flattend_indices * torch.unsqueeze(flattened_indices_dim, dim=0)
            flattend_indices = flattend_indices.sum(dim=-1)

            # accumulates partial results from filter positions in output locations O = SUM_i( I_i * W_i )
            # transform view to 1d
            output_feature_map = output_feature_map.view([-1, self.nOut])
            # add computed update terms to the output feature map at the respective output locations
            # .index_add_ might not work if gradients are needed
            output_feature_map.index_add_(dim=0, index=flattend_indices, source=update_term)
            # restore view to 2d
            output_feature_map = output_feature_map.view(spatial_dimension_out + [self.nOut])

        # Set deactivated update sites in the output to zero, but keep it in the rulebook for the next layers
        output_feature_map = self.set_new_inactive_sites(active_sites_map, output_feature_map)

        print_debug("adding bias.")

        # add bias to complete filter application
        if self.use_bias:
            # TODO handle stride
            output_feature_map[active_sites_map == Sites.NEW_ACTIVE_SITE.value] += self.bias

        print_debug("cleanup.")

        # help with resource management
        del self.old_input_feature_map
        del self.output_feature_map
        # make deepcopies of current input and output feature map for use in future method calls
        self.old_input_feature_map = feature_map.clone()
        self.output_feature_map = output_feature_map.clone()

        print_debug("done.")

        return new_update_events, output_feature_map, active_sites_map, rule_book_input, rule_book_output

    def updateRuleBooks(self, active_sites_map, update_location, bool_new_active_site, zero_input_update,
                        rule_book_input, rule_book_output, update_location_indices):
        """
        Updates the rule books used for the weight multiplication
        :param active_sites_map: map indicating for each spatial location in the input tensor whether this site is
                                 active
        :param update_location: list of spatial locations that are updated by events
        :param bool_new_active_site: map indicating for each spatial location in the input tensor whether this site is
                                     newly active
        :param zero_input_update: list of update locations that have the value 0 in feature_map
        :param rule_book_input: input-rulebook of previous layer
        :param rule_book_output: output-rulebook of previous layer
        :param update_location_indices: tuple of individual 2d-location in update_location
        :return: updated rulebooks, newly activated sites that just generated an internal 'event' and need to be
                 passed to the following layer, updated active_sites_map
        """

        # Pad input to index with kernel
        padded_active_sites_map = torch.nn.functional.pad(active_sites_map, self.padding, mode='constant', value=0)
        shifted_update_location = update_location + (self.filter_size_tensor // 2)[None, :]

        # Compute indices corresponding to the receptive fields of the update location
        kernel_update_location = (
                shifted_update_location[:, None, :]
                + (self.kernel_indices[None, :, :] - self.filter_size // 2)  # convert indices to offset
        )

        active_sites_to_update = self.compute_active_sites_to_update(padded_active_sites_map, kernel_update_location)

        active_sites_map = self.mark_updated_sites(active_sites_map, update_location_indices)

        if self.first_layer:  # zero_input_update is not None:  # TODO
            # Set new deactivate sites to Sites.NEW_INACTIVE_SITE
            active_sites_map[update_location[zero_input_update].split(1, dim=-1)] = Sites.NEW_INACTIVE_SITE.value

        new_update_events = []

        # relative location (negative 1d offset) of an active site inside the kernel around an update location
        # (linked by following input_locations)
        position_kernels = self.filter_volume - 1 - active_sites_to_update[:, 1]
        # input location for the respective relative output location at same position in position_kernels
        input_locations = update_location[active_sites_to_update[:, 0], :].clone()
        # position_kernels converted to negative 2d offsets inside kernel
        nd_kernel_positions = self.kernel_indices[position_kernels]
        # actual output locations for the respective input locations in input_locations
        # subtract each negative 2d offset from highest location in kernel around respective input location
        output_locations = input_locations + self.filter_size // 2 - nd_kernel_positions
        # output locations as seperate x and y coordinates
        output_location_indices_i, output_location_indices_j = output_locations.split(1, dim=-1)
        # retrieve current Site Status for each output active site
        output_active_sites = active_sites_map[output_locations[..., 0], output_locations[..., 1]]

        # Compute Rule Book

        # find output active sites that are neither new active nor new inactive
        #   list all output site indices
        i_active_sites = torch.arange(active_sites_to_update.shape[0], dtype=torch.long)
        #   filter out new active and new inactive output sites
        i_active_sites = self.filter_new_status_sites(i_active_sites, output_active_sites)
        print_debug("iterating active sites:")
        # process each such site:
        for i_active_site in i_active_sites:
            print_debug(f"processing active site {i_active_site + 1}/{len(i_active_sites)}.", 1)
            # retrieve current site status
            output_active_site = active_sites_map[output_location_indices_i[i_active_site],
                                                  output_location_indices_j[i_active_site]]

            # append rule
            rule_book_output[position_kernels[i_active_site]].append(output_locations[i_active_site])
            rule_book_input[position_kernels[i_active_site]].append(input_locations[i_active_site])

            # if the site was not yet updated
            if self.update_condition(output_active_site):
                # mark it as new active for the next layer
                new_update_events.append(self.convert_output_locations(output_locations[i_active_site]))
                # TODO the same?
                # and mark it as updated
                active_sites_map[output_location_indices_i[i_active_site],
                                 output_location_indices_j[i_active_site]] = Sites.UPDATED_SITE.value
                active_sites_map[output_locations[i_active_site, ..., 0],
                                output_locations[i_active_site, ..., 1]] = Sites.UPDATED_SITE.value

        # Set newly initialised sites to 3 equal to Sites.NEW_ACTIVE_SITE
        active_sites_map[update_location[bool_new_active_site].split(1, dim=-1)] = Sites.NEW_ACTIVE_SITE.value
        if self.need_to_reverse_newly_active_sites():
            # "Update neuron if it is first time active. Exclude points, which influence is propagated at same time step"
            # -> Update sites if they are new active sites and have not already been updated by the influence of normal
            # active sites in the above block of code
            #   exclude already updated sites by zeroing them in the purely local variable padded_active_sites_map
            padded_active_sites_map[shifted_update_location.split(1, dim=-1)] = 0
            # #   allow subclasses to apply additional filters
            # self.filter_output_sites(padded_active_sites_map)
            # "Update the influence from the active sites in the receptive field, if the site is newly active"
            if bool_new_active_site.nelement() != 0:  # TODO
                # Return if no new activate site are given as input
                if bool_new_active_site.sum() == 0:
                    return rule_book_input, rule_book_output, new_update_events, active_sites_map
                # compute new active site influences analogously to active_sites_to_update
                # (only considering kernels around newly active sites)
                new_active_sites_influence = self.compute_active_sites_to_update(
                    padded_active_sites_map,
                    kernel_update_location[bool_new_active_site, :],
                    reverse=True
                )
                print_debug("reversing influence of newly active sites:")
                # process each new active site influence
                # (one influence at a time, a single newly active site might be processed in several iterations of the loop)
                for i_new_active_site in range(new_active_sites_influence.shape[0]):
                    print_debug(f"processing active site {i_new_active_site + 1}/{new_active_sites_influence.shape[0]}.", 1)
                    # this site has not yet been updated, treat it as output location for the additional rules
                    # get the influence 1d offset inside the kernel
                    position_kernel = new_active_sites_influence[i_new_active_site, 1]
                    # get real position of respective new active site, use as output location
                    output_location = self.get_output_location(
                        update_locations=update_location[bool_new_active_site],
                        index=new_active_sites_influence[i_new_active_site, 0]
                    )
                    # get real position of the site influencing the current new active site, use as input location
                    input_location = output_location - self.filter_size // 2 + self.kernel_indices[position_kernel]
                    # append rules
                    rule_book_output[position_kernel].append(output_location.clone())
                    rule_book_input[position_kernel].append(input_location)

        print_debug("rulebook update complete.")

        return rule_book_input, rule_book_output, new_update_events, active_sites_map

    def checkInputArguments(self, update_location, feature_map, active_sites_map):
        """Checks if the input arguments have the correct shape"""
        if update_location.ndim != 2 or update_location.shape[-1] != self.dimension:
            raise ValueError('Expected update_location to have shape [N, %s]. Got size %s' %
                             (self.dimension, list(update_location.shape)))
        if feature_map.ndim != self.dimension + 1 or feature_map.shape[-1] != self.nIn:
            raise ValueError('Expected feature_map to have shape [Spatial_1, Spatial_2, ..., %s]. Got size %s' %
                             (self.nIn, list(feature_map.shape)))
        if active_sites_map is None:
            return
        if active_sites_map.ndim != self.dimension:
            raise ValueError('Expected active_sites_map to have %s dimensions. Got size %s' %
                             (self.dimension, list(active_sites_map.shape)))


class asynSparseConvolution2D(asynSparseConvolution2DBase):
    """
    "Valid Sparse Convolution (VSC)" or "Submanifold Sparse Convolution" (kept old class name for backward compatability)
    """
    def requires_new_rulebooks(self):
        return self.first_layer

    def get_spatial_dimension_out(self, feature_map):
        return self.get_spatial_dimension_in(feature_map)

    def filter_zero_update_locations(self, update_locations, feature_map):
        return update_locations

    def set_new_inactive_sites(self, active_sites_map, output_feature_map):
        output_feature_map = output_feature_map * \
                             torch.unsqueeze((active_sites_map != Sites.NEW_INACTIVE_SITE.value).float(), -1)
        return output_feature_map

    def filter_new_status_sites(self, i_active_sites, output_active_sites):
        return i_active_sites[
            (output_active_sites != Sites.NEW_ACTIVE_SITE.value)
            &
            (output_active_sites != Sites.NEW_INACTIVE_SITE.value)
        ]

    def mark_updated_sites(self, active_sites_map, update_location_indices):
        # # Set updated sites to 2
        active_sites_map[update_location_indices] = Sites.UPDATED_SITE.value
        return active_sites_map

    def update_condition(self, output_active_site):
        # return output_active_site != Sites.UPDATED_SITE.value
        return output_active_site == Sites.ACTIVE_SITE.value  # TODO

    def filter_newly_active_sites(self, active_sites_map, input_indices, output_indices):
        return (active_sites_map[output_indices] != Sites.NEW_ACTIVE_SITE.value).float()

    def need_to_reverse_newly_active_sites(self):
        return True

    def forward(self, update_location, feature_map, active_sites_map=None, rule_book_input=None, rule_book_output=None):
        # overload to be able to differentiate between calls by different layer types in profiler
        return super().forward(
            update_location, feature_map, active_sites_map=active_sites_map, rule_book_input=rule_book_input,
            rule_book_output=rule_book_output
        )

    def __str__(self):
        return f"ValidSparseConvolution2D{'*' if self.first_layer else ''} {self.nIn}->{self.nOut} C{self.filter_size}"


class asynNonValidSparseConvolution2D(asynSparseConvolution2DBase):
    """
    "(non-valid) Sparse Convolution (SC)" or "Sparse Convolution"
    """
    def __init__(self, dimension, nIn, nOut, filter_size, filter_stride=1, first_layer=False, use_bias=False,
                 device=torch.device('cpu')):
        super().__init__(dimension, nIn, nOut, filter_size, first_layer=first_layer, use_bias=use_bias,
             device=device)

        self.stride = filter_stride
        # prepare filter for sites matching the stride
        self.stride_filter = torch.stack(
            torch.meshgrid(
                [torch.arange(self.stride) for _ in range(2)]
            ),
            dim=-1
        )

    def get_spatial_dimension_out(self, feature_map):
        return [(size + self.stride - 1) // self.stride for size in feature_map.shape[:-1]]

    def requires_new_rulebooks(self):
        return self.first_layer or self.stride > 1

    def filter_zero_update_locations(self, update_locations, feature_map):
        update_location_indices = update_locations.split(1, dim=-1)
        zero_input_update = torch.squeeze(torch.sum(feature_map[update_location_indices] ** 2, dim=-1) == 0, dim=-1)
        return update_locations[~zero_input_update]

    def set_new_inactive_sites(self, active_sites_map, output_feature_map):
        # get stride filter
        # TODO check
        # active_sites_map_filter = torch.meshgrid(
        #     torch.arange(0, active_sites_map.shape[0], self.stride),
        #     torch.arange(0, active_sites_map.shape[1], self.stride)
        # )
        # output_feature_map = (
        #     output_feature_map *
        #     torch.unsqueeze((active_sites_map[active_sites_map_filter] != Sites.NEW_INACTIVE_SITE.value).float(), -1)
        # )
        return output_feature_map

    def filter_new_status_sites(self, i_active_sites, output_active_sites):
        # return i_active_sites[
        #     (output_active_sites != Sites.NEW_INACTIVE_SITE.value)
        #     &
        #     (output_active_sites != Sites.NEW_ACTIVE_SITE.value)
        # ]  # TODO
        return i_active_sites

    def need_to_reverse_newly_active_sites(self):
        return False

    def compute_active_sites_to_update(self, padded_active_sites_map, kernel_update_location, reverse=False):
        if reverse:
            # finding input sites, center of kernels need to match stride
            return super().compute_active_sites_to_update(
                padded_active_sites_map,
                # match stride
                kernel_update_location[
                    # if center of (stratified) kernel does not match stride, dismiss update location
                    # kernel_update_location.shape: [no_locations, self.filter_volume, 2]
                    (
                        # brackets around next line are important!
                        (kernel_update_location[:, self.filter_volume // 2, :] - self.filter_size // 2)
                        % self.stride == 0
                    ).all(axis=-1)
                ]
            )
        else:  # find output sites, need to match stride and bounds
            # dismiss sites that do not match the stride
            site_matches_stride = (kernel_update_location - self.filter_size // 2) % self.stride == 0
            # dismiss sites in the padded rim of the padded_active_sites_map
            site_within_bounds = torch.logical_and(
                self.filter_size // 2 <= kernel_update_location,
                kernel_update_location < torch.Tensor(list(padded_active_sites_map.shape)) - (self.filter_size // 2)
            )
            # combine both filters
            active_sites_to_update = (
                    torch.logical_and(
                        site_matches_stride,
                        site_within_bounds
                    )
            ).all(axis=-1)
            # convert to coordinates
            active_sites_to_update = active_sites_to_update.nonzero()
            return active_sites_to_update

    def get_output_location(self, update_locations, index):
        # filter update_locations analogously to compute_active_sites_to_update(..., reverse=True)
        # important: update_locations are not padded, kernel locations in
        # compute_active_sites_to_update(..., reverse=True) were
        return super().get_output_location(
            update_locations[(update_locations % self.stride == 0).all(axis=-1)],
            index
        )

    def filter_output_sites(self, padded_active_sites_map):
        # repeat stride filter to size of padded_active_sites_map
        stride_index = self.stride_filter.repeat(
            [size // self.stride + 1 for size in padded_active_sites_map.shape] + [1]
        )[:padded_active_sites_map.shape[0], :padded_active_sites_map.shape[1], :]
        # set filter to everything except sites matching the stride
        padded_active_sites_map_filter = (
                stride_index - (self.filter_size // 2) % self.stride != 0
        ).any(axis=-1)
        # deactivate all sites not matching the stride
        padded_active_sites_map[padded_active_sites_map_filter] = 0
        return padded_active_sites_map

    def mark_updated_sites(self, active_sites_map, update_location_indices):
        # pass
        active_sites_map[update_location_indices] = Sites.UPDATED_SITE.value  # TODO
        return active_sites_map

    def update_condition(self, output_active_site):
        return output_active_site != Sites.UPDATED_SITE.value and output_active_site != Sites.NEW_INACTIVE_SITE.value
        # return output_active_site == Sites.ACTIVE_SITE.value

    def filter_newly_active_sites(self, active_sites_map, input_indices, output_indices):
        return (active_sites_map[input_indices] != Sites.NEW_ACTIVE_SITE.value).float()
        # return 1

    def convert_output_locations(self, output_locations):
        assert (output_locations % self.stride == 0).all()  # TODO
        return (output_locations / self.stride).long()

    def forward(self, update_location, feature_map, active_sites_map=None, rule_book_input=None, rule_book_output=None):
        out = super().forward(
            update_location, feature_map, active_sites_map=active_sites_map, rule_book_input=rule_book_input,
            rule_book_output=rule_book_output
        )
        new_update_events, output_feature_map, active_sites_map, rule_book_input, rule_book_output = out
        if self.stride > 1:
            # suppress rulebooks and subsample active sites map, as they are specific for this layer
            rule_book_input = None
            rule_book_output = None
            active_sites_map_filter = torch.meshgrid(
                torch.arange(0, active_sites_map.shape[0], self.stride),
                torch.arange(0, active_sites_map.shape[1], self.stride)
            )
            active_sites_map = active_sites_map[active_sites_map_filter]
        return (
            new_update_events.long(), output_feature_map, active_sites_map,
            rule_book_input, rule_book_output
        )

    def updateRuleBooks(self, active_sites_map, update_location, bool_new_active_site, zero_input_update,
                        rule_book_input, rule_book_output, update_location_indices):
        # filter to only update locations lying in the receptive field of a location matching the stride

        # active sites may only be ignored if stride is greater filter size, else the convolution always covers
        # the whole input tensor
        if self.stride > self.filter_size:
            # due to same padding (following instruction block), first location matching the stride is (0, 0) (for now)
            # assuming filter_size is uneven
            # -> (location % stride) + (filter_size // 2) < filter_size  # has to hold to keep update location
            keep_update_locations = torch.all(
                torch.stack(
                    [
                        (location % self.stride) + (self.filter_size // 2) < self.filter_size
                        for location
                        in list(update_location)
                    ]
                ),
                dim=-1
            )
            update_location = update_location[keep_update_locations]
            zero_input_update = zero_input_update[keep_update_locations]
            bool_new_active_site = bool_new_active_site[keep_update_locations]
        # call base method with updated 'update_location'
        return super().updateRuleBooks(active_sites_map, update_location, bool_new_active_site, zero_input_update,
                                       rule_book_input, rule_book_output, update_location_indices)

    def __str__(self):
        return f"AsynNonValidSparseConvolution2D{'*' if self.first_layer else ''} {self.nIn}->{self.nOut} " \
               f"C{self.filter_size}/{self.stride}"

