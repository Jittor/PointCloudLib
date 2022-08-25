from random import sample
import jittor as jt
# OS functions
from os import listdir
from os.path import exists, join
import numpy as np
import time
import pickle

import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def create_3D_rotations(axis, angle):
    """
    Create rotation matrices from a list of axes and angles. Code from wikipedia on quaternions
    :param axis: float32[N, 3]
    :param angle: float32[N,]
    :return: float32[N, 3, 3]
    """

    t1 = np.cos(angle)
    t2 = 1 - t1
    t3 = axis[:, 0] * axis[:, 0]
    t6 = t2 * axis[:, 0]
    t7 = t6 * axis[:, 1]
    t8 = np.sin(angle)
    t9 = t8 * axis[:, 2]
    t11 = t6 * axis[:, 2]
    t12 = t8 * axis[:, 1]
    t15 = axis[:, 1] * axis[:, 1]
    t19 = t2 * axis[:, 1] * axis[:, 2]
    t20 = t8 * axis[:, 0]
    t24 = axis[:, 2] * axis[:, 2]
    R = np.stack([t1 + t2 * t3,
                  t7 - t9,
                  t11 + t12,
                  t7 + t9,
                  t1 + t2 * t15,
                  t19 - t20,
                  t11 - t12,
                  t19 + t20,
                  t1 + t2 * t24], axis=1)

    return np.reshape(R, (-1, 3, 3))

def batch_grid_subsampling(points, batches_len, features=None, labels=None,
                           sampleDl=0.1, max_p=0, verbose=0, random_grid_orient=True):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    R = None
    B = len(batches_len)
    if random_grid_orient:

        ########################################################
        # Create a random rotation matrix for each batch element
        ########################################################

        # Choose two random angles for the first vector in polar coordinates
        theta = np.random.rand(B) * 2 * np.pi
        phi = (np.random.rand(B) - 0.5) * np.pi

        # Create the first vector in carthesian coordinates
        u = np.vstack([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

        # Choose a random rotation angle
        alpha = np.random.rand(B) * 2 * np.pi

        # Create the rotation matrix with this vector and angle
        R = create_3D_rotations(u.T, alpha).astype(np.float32)

        #################
        # Apply rotations
        #################

        i0 = 0
        points = points.copy()
        for bi, length in enumerate(batches_len):
            # Apply the rotation
            points[i0:i0 + length, :] = np.sum(np.expand_dims(points[i0:i0 + length, :], 2) * R[bi], axis=1)
            i0 += length

    #######################
    # Sunsample and realign
    #######################

    if (features is None) and (labels is None):
        s_points, s_len = cpp_subsampling.subsample_batch(points,
                                                          batches_len,
                                                          sampleDl=sampleDl,
                                                          max_p=max_p,
                                                          verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len

    elif (labels is None):
        s_points, s_len, s_features = cpp_subsampling.subsample_batch(points,
                                                                      batches_len,
                                                                      features=features,
                                                                      sampleDl=sampleDl,
                                                                      max_p=max_p,
                                                                      verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len, s_features

    elif (features is None):
        s_points, s_len, s_labels = cpp_subsampling.subsample_batch(points,
                                                                    batches_len,
                                                                    classes=labels,
                                                                    sampleDl=sampleDl,
                                                                    max_p=max_p,
                                                                    verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len, s_labels

    else:
        s_points, s_len, s_features, s_labels = cpp_subsampling.subsample_batch(points,
                                                                              batches_len,
                                                                              features=features,
                                                                              classes=labels,
                                                                              sampleDl=sampleDl,
                                                                              max_p=max_p,
                                                                              verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len, s_features, s_labels

def batch_neighbors(queries, supports, q_batches, s_batches, radius):
    """
    Computes neighbors for a batch of queries and supports
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B)the list of lengths of batch elements in supports
    :param radius: float32
    :return: neighbors indices
    """

    return cpp_neighbors.batch_query(queries, supports, q_batches, s_batches, radius=radius)


class ModelNet40Dataset():
    """Class to handle Modelnet 40 dataset."""

    def __init__(self, config, train=True, orient_correction=True):
        """
        This dataset is small enough to be stored in-memory, so load all point clouds here
        """

        self.neighborhood_limits = []
        ############
        # Parameters
        ############

        # Dict from labels to names
        self.label_to_names = {0: 'airplane',
                               1: 'bathtub',
                               2: 'bed',
                               3: 'bench',
                               4: 'bookshelf',
                               5: 'bottle',
                               6: 'bowl',
                               7: 'car',
                               8: 'chair',
                               9: 'cone',
                               10: 'cup',
                               11: 'curtain',
                               12: 'desk',
                               13: 'door',
                               14: 'dresser',
                               15: 'flower_pot',
                               16: 'glass_box',
                               17: 'guitar',
                               18: 'keyboard',
                               19: 'lamp',
                               20: 'laptop',
                               21: 'mantel',
                               22: 'monitor',
                               23: 'night_stand',
                               24: 'person',
                               25: 'piano',
                               26: 'plant',
                               27: 'radio',
                               28: 'range_hood',
                               29: 'sink',
                               30: 'sofa',
                               31: 'stairs',
                               32: 'stool',
                               33: 'table',
                               34: 'tent',
                               35: 'toilet',
                               36: 'tv_stand',
                               37: 'vase',
                               38: 'wardrobe',
                               39: 'xbox'}

        # Initialize a bunch of variables concerning class labels
        self.init_labels()

        # List of classes ignored during training (can be empty)
        self.ignored_labels = np.array([])

        # Dataset folder
        self.path = '../Data/ModelNet40'

        # Type of task conducted on this dataset
        self.dataset_task = 'classification'

        # Update number of class and data task in configuration
        config.num_classes = self.num_classes
        config.dataset_task = self.dataset_task

        # Parameters from config
        self.config = config

        # Training or test set
        self.train = train

        # Number of models and models used per epoch
        if self.train:
            self.num_models = 9843
            if config.epoch_steps and config.epoch_steps * config.batch_num < self.num_models:
                self.epoch_n = config.epoch_steps * config.batch_num
            else:
                self.epoch_n = self.num_models
        else:
            self.num_models = 2468
            self.epoch_n = min(self.num_models, config.validation_size * config.batch_num)

        #############
        # Load models
        #############

        if 0 < self.config.first_subsampling_dl <= 0.01:
            raise ValueError('subsampling_parameter too low (should be over 1 cm')

        self.input_points, self.input_normals, self.input_labels = self.load_subsampled_clouds(orient_correction)

        return

    def __len__(self):
        """
        Return the length of data here
        """
        return self.num_models

    def __getitem__(self, idx_list):

        ###################
        # Gather batch data
        ###################

        if isinstance(idx_list, int):
            idx_list = [idx_list]

        tp_list = []
        tn_list = []
        tl_list = []
        ti_list = []
        s_list = []
        R_list = []

        for p_i in idx_list:
            # Get points and labels
            points = self.input_points[p_i].astype(np.float32)
            normals = self.input_normals[p_i].astype(np.float32)
            label = self.label_to_idx[self.input_labels[p_i]]

            # Data augmentation
            points, normals, scale, R = self.augmentation_transform(points, normals)

            # Stack batch
            tp_list += [points]
            tn_list += [normals]
            tl_list += [label]
            ti_list += [p_i]
            s_list += [scale]
            R_list += [R]

        ###################
        # Concatenate batch
        ###################

        #show_ModelNet_examples(tp_list, cloud_normals=tn_list)

        stacked_points = np.concatenate(tp_list, axis=0)
        stacked_normals = np.concatenate(tn_list, axis=0)
        labels = np.array(tl_list, dtype=np.int64)
        model_inds = np.array(ti_list, dtype=np.int32)
        stack_lengths = np.array([tp.shape[0] for tp in tp_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(R_list, axis=0)

        # Input features
        stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
        if self.config.in_features_dim == 1:
            pass
        elif self.config.in_features_dim == 4:
            stacked_features = np.hstack((stacked_features, stacked_normals))
        else:
            raise ValueError('Only accepted input dimensions are 1, 4 and 7 (without and with XYZ)')

        #######################
        # Create network inputs
        #######################
        #
        #   Points, neighbors, pooling indices for each layers
        #

        # Get the whole input list
        input_list = self.classification_inputs(stacked_points,
                                                stacked_features,
                                                labels,
                                                stack_lengths)

        # Add scale and rotation for testing
        input_list += [scales, rots, model_inds]

        return input_list

    def init_labels(self):

        # Initialize all label parameters given the label_to_names dict
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_names = [self.label_to_names[k] for k in self.label_values]
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}

    def load_subsampled_clouds(self, orient_correction):

        # Restart timer
        t0 = time.time()

        # Load wanted points if possible
        if self.train:
            split ='training'
        else:
            split = 'test'

        print('\nLoading {:s} points subsampled at {:.3f}'.format(split, self.config.first_subsampling_dl))
        filename = join(self.path, '{:s}_{:.3f}_record.pkl'.format(split, self.config.first_subsampling_dl))
        print("trying to load data from ", filename)
        if exists(filename):
            with open(filename, 'rb') as file:
                input_points, input_normals, input_labels = pickle.load(file)
        # Else compute them from original points
        else:
            # Collect training file names
            print("subsampled_clouds not loaded, exit!")
            exit(0)

        lengths = [p.shape[0] for p in input_points]
        sizes = [l * 4 * 6 for l in lengths]
        print('{:.1f} MB loaded in {:.1f}s'.format(np.sum(sizes) * 1e-6, time.time() - t0))

        if orient_correction:
            input_points = [pp[:, [0, 2, 1]] for pp in input_points]
            input_normals = [nn[:, [0, 2, 1]] for nn in input_normals]

        return input_points, input_normals, input_labels

    def augmentation_transform(self, points, normals=None, verbose=False):
        """Implementation of an augmentation transform for point clouds."""

        ##########
        # Rotation
        ##########

        # Initialize rotation matrix
        R = np.eye(points.shape[1])

        if points.shape[1] == 3:
            if self.config.augment_rotation == 'vertical':

                # Create random rotations
                theta = np.random.rand() * 2 * np.pi
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

            elif self.config.augment_rotation == 'all':

                # Choose two random angles for the first vector in polar coordinates
                theta = np.random.rand() * 2 * np.pi
                phi = (np.random.rand() - 0.5) * np.pi

                # Create the first vector in carthesian coordinates
                u = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

                # Choose a random rotation angle
                alpha = np.random.rand() * 2 * np.pi

                # Create the rotation matrix with this vector and angle
                R = create_3D_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[0]

        R = R.astype(np.float32)

        #######
        # Scale
        #######

        # Choose random scales for each example
        min_s = self.config.augment_scale_min
        max_s = self.config.augment_scale_max
        if self.config.augment_scale_anisotropic:
            scale = np.random.rand(points.shape[1]) * (max_s - min_s) + min_s
        else:
            scale = np.random.rand() * (max_s - min_s) + min_s

        # Add random symmetries to the scale factor
        symmetries = np.array(self.config.augment_symmetries).astype(np.int32)
        symmetries *= np.random.randint(2, size=points.shape[1])
        scale = (scale * (1 - symmetries * 2)).astype(np.float32)

        #######
        # Noise
        #######

        noise = (np.random.randn(points.shape[0], points.shape[1]) * self.config.augment_noise).astype(np.float32)

        ##################
        # Apply transforms
        ##################

        # Do not use np.dot because it is multi-threaded
        #augmented_points = np.dot(points, R) * scale + noise
        augmented_points = np.sum(np.expand_dims(points, 2) * R, axis=1) * scale + noise


        if normals is None:
            return augmented_points, scale, R
        else:
            # Anisotropic scale of the normals thanks to cross product formula
            normal_scale = scale[[1, 2, 0]] * scale[[2, 0, 1]]
            augmented_normals = np.dot(normals, R) * normal_scale
            # Renormalise
            augmented_normals *= 1 / (np.linalg.norm(augmented_normals, axis=1, keepdims=True) + 1e-6)

            if verbose:
                test_p = [np.vstack([points, augmented_points])]
                test_n = [np.vstack([normals, augmented_normals])]
                test_l = [np.hstack([points[:, 2]*0, augmented_points[:, 2]*0+1])]
                show_ModelNet_examples(test_p, test_n, test_l)

            return augmented_points, augmented_normals, scale, R

    def classification_inputs(self,
                              stacked_points,
                              stacked_features,
                              labels,
                              stack_lengths):

        # Starting radius of convolutions
        r_normal = self.config.first_subsampling_dl * self.config.conv_radius

        # Starting layer
        layer_blocks = []

        # Lists of inputs
        input_points = []
        input_neighbors = []
        input_pools = []
        input_stack_lengths = []
        deform_layers = []

        ######################
        # Loop over the blocks
        ######################

        arch = self.config.architecture

        for block_i, block in enumerate(arch):

            # Get all blocks of the layer
            if not ('pool' in block or 'strided' in block or 'global' in block or 'upsample' in block):
                layer_blocks += [block]
                continue

            # Convolution neighbors indices
            # *****************************

            deform_layer = False
            if layer_blocks:
                # Convolutions are done in this layer, compute the neighbors with the good radius
                if np.any(['deformable' in blck for blck in layer_blocks]):
                    r = r_normal * self.config.deform_radius / self.config.conv_radius
                    deform_layer = True
                else:
                    r = r_normal
                conv_i = batch_neighbors(stacked_points, stacked_points, stack_lengths, stack_lengths, r)

            else:
                # This layer only perform pooling, no neighbors required
                conv_i = np.zeros((0, 1), dtype=np.int32)

            # Pooling neighbors indices
            # *************************

            # If end of layer is a pooling operation
            if 'pool' in block or 'strided' in block:

                # New subsampling length
                dl = 2 * r_normal / self.config.conv_radius

                # Subsampled points
                pool_p, pool_b = batch_grid_subsampling(stacked_points, stack_lengths, sampleDl=dl)

                # Radius of pooled neighbors
                if 'deformable' in block:
                    r = r_normal * self.config.deform_radius / self.config.conv_radius
                    deform_layer = True
                else:
                    r = r_normal

                # Subsample indices
                pool_i = batch_neighbors(pool_p, stacked_points, pool_b, stack_lengths, r)

            else:
                # No pooling in the end of this layer, no pooling indices required
                pool_i = np.zeros((0, 1), dtype=np.int32)
                pool_p = np.zeros((0, 1), dtype=np.float32)
                pool_b = np.zeros((0,), dtype=np.int32)

            # Reduce size of neighbors matrices by eliminating furthest point
            conv_i = self.big_neighborhood_filter(conv_i, len(input_points))
            pool_i = self.big_neighborhood_filter(pool_i, len(input_points))

            # Updating input lists
            input_points += [stacked_points]
            input_neighbors += [conv_i.astype(np.int64)]
            input_pools += [pool_i.astype(np.int64)]
            input_stack_lengths += [stack_lengths]
            deform_layers += [deform_layer]

            # New points for next layer
            stacked_points = pool_p
            stack_lengths = pool_b

            # Update radius and reset blocks
            r_normal *= 2
            layer_blocks = []

            # Stop when meeting a global pooling or upsampling
            if 'global' in block or 'upsample' in block:
                break

        ###############
        # Return inputs
        ###############

        # Save deform layers

        # list of network inputs
        li = input_points + input_neighbors + input_pools + input_stack_lengths
        li += [stacked_features, labels]

        return li

    def big_neighborhood_filter(self, neighbors, layer):
        """
        Filter neighborhoods with max number of neighbors. Limit is set to keep XX% of the neighborhoods untouched.
        Limit is computed at initialization
        """

        # crop neighbors matrix
        if len(self.neighborhood_limits) > 0:
            return neighbors[:, :self.neighborhood_limits[layer]]
        else:
            return neighbors


class ModelNet40Sampler():
    """Sampler for ModelNet40"""

    def __init__(self, dataset: ModelNet40Dataset, use_potential=True, balance_labels=False):

        # Does the sampler use potential for regular sampling
        self.use_potential = use_potential

        # Should be balance the classes when sampling
        self.balance_labels = balance_labels

        # Dataset used by the sampler (no copy is made in memory)
        self.dataset = dataset

        # Create potentials
        if self.use_potential:
            self.potentials = np.random.rand(len(dataset.input_labels)) * 0.1 + 0.1
        else:
            self.potentials = None

        # Initialize value for batch limit (max number of points per batch).
        self.batch_limit = 10000

        return

    def __iter__(self):
        """
        Yield next batch indices here
        """

        ##########################################
        # Initialize the list of generated indices
        ##########################################

        if self.use_potential:
            if self.balance_labels:

                gen_indices = []
                pick_n = self.dataset.epoch_n // self.dataset.num_classes + 1
                for i, l in enumerate(self.dataset.label_values):

                    # Get the potentials of the objects of this class
                    label_inds = np.where(np.equal(self.dataset.input_labels, l))[0]
                    class_potentials = self.potentials[label_inds]

                    # Get the indices to generate thanks to potentials
                    if pick_n < class_potentials.shape[0]:
                        pick_indices = np.argpartition(class_potentials, pick_n)[:pick_n]
                    else:
                        pick_indices = np.random.permutation(class_potentials.shape[0])
                    class_indices = label_inds[pick_indices]
                    gen_indices.append(class_indices)

                # Stack the chosen indices of all classes
                gen_indices = np.random.permutation(np.hstack(gen_indices))

            else:

                # Get indices with the minimum potential
                if self.dataset.epoch_n < self.potentials.shape[0]:
                    gen_indices = np.argpartition(self.potentials, self.dataset.epoch_n)[:self.dataset.epoch_n]
                else:
                    gen_indices = np.random.permutation(self.potentials.shape[0])
                gen_indices = np.random.permutation(gen_indices)

            # Update potentials (Change the order for the next epoch)
            self.potentials[gen_indices] = np.ceil(self.potentials[gen_indices])
            self.potentials[gen_indices] += np.random.rand(gen_indices.shape[0]) * 0.1 + 0.1

        else:
            if self.balance_labels:
                pick_n = self.dataset.epoch_n // self.dataset.num_classes + 1
                gen_indices = []
                for l in self.dataset.label_values:
                    label_inds = np.where(np.equal(self.dataset.input_labels, l))[0]
                    rand_inds = np.random.choice(label_inds, size=pick_n, replace=True)
                    gen_indices += [rand_inds]
                gen_indices = np.random.permutation(np.hstack(gen_indices))
            else:
                gen_indices = np.random.permutation(self.dataset.num_models)[:self.dataset.epoch_n]

        ################
        # Generator loop
        ################

        # Initialize concatenation lists
        ti_list = []
        batch_n = 0

        # Generator loop
        for p_i in gen_indices:
        # for s, x in enumerate(self.dataset.input_points):
            # Size of picked cloud
            n = self.dataset.input_points[p_i].shape[0]

            # In case batch is full, yield it and reset it
            if batch_n + n > self.batch_limit and batch_n > 0:
                yield np.array(ti_list, dtype=np.int32)
                ti_list = []
                batch_n = 0

            # Add data to current batch
            ti_list += [p_i]

            # Update batch size
            batch_n += n

        yield np.array(ti_list, dtype=np.int32)

        return 0

    def __len__(self):
        """
        The number of yielded samples is variable
        """
        return None

    def calibration(self, verbose=False):
        """
        Method performing batch and neighbors calibration.
            Batch calibration: Set "batch_limit" (the maximum number of points allowed in every batch) so that the
                               average batch size (number of stacked pointclouds) is the one asked.
        Neighbors calibration: Set the "neighborhood_limits" (the maximum number of neighbors allowed in convolutions)
                               so that 90% of the neighborhoods remain untouched. There is a limit for each layer.
        """

        ##############################
        # Previously saved calibration
        ##############################

        print('\nStarting Calibration (use verbose=True for more details)')
        t0 = time.time()

        redo = False

        # Batch limit
        # ***********

        # Load batch_limit dictionary
        batch_lim_file = join(self.dataset.path, 'batch_limits.pkl')
        if exists(batch_lim_file):
            with open(batch_lim_file, 'rb') as file:
                batch_lim_dict = pickle.load(file)
        else:
            batch_lim_dict = {}

        # Check if the batch limit associated with current parameters exists
        key = '{:.3f}_{:d}'.format(self.dataset.config.first_subsampling_dl,
                                   self.dataset.config.batch_num)
        if key in batch_lim_dict:
            self.batch_limit = batch_lim_dict[key]
            self.batch_limit //= 4
            print("batch_limit: ", self.batch_limit)
        else:
            redo = True

        if verbose:
            print('\nPrevious calibration found:')
            print('Check batch limit dictionary')
            if key in batch_lim_dict:
                color = bcolors.OKGREEN
                v = str(int(batch_lim_dict[key]))
            else:
                color = bcolors.FAIL
                v = '?'
            print('{:}\"{:s}\": {:s}{:}'.format(color, key, v, bcolors.ENDC))

        # Neighbors limit
        # ***************

        # Load neighb_limits dictionary
        neighb_lim_file = join(self.dataset.path, 'neighbors_limits.pkl')
        if exists(neighb_lim_file):
            with open(neighb_lim_file, 'rb') as file:
                neighb_lim_dict = pickle.load(file)
        else:
            neighb_lim_dict = {}

        # Check if the limit associated with current parameters exists (for each layer)
        neighb_limits = []
        for layer_ind in range(self.dataset.config.num_layers):

            dl = self.dataset.config.first_subsampling_dl * (2**layer_ind)
            if self.dataset.config.deform_layers[layer_ind]:
                r = dl * self.dataset.config.deform_radius
            else:
                r = dl * self.dataset.config.conv_radius

            key = '{:.3f}_{:.3f}'.format(dl, r)
            if key in neighb_lim_dict:
                neighb_limits += [neighb_lim_dict[key]]

        if len(neighb_limits) == self.dataset.config.num_layers:
            self.dataset.neighborhood_limits = neighb_limits
        else:
            redo = True

        if verbose:
            print('Check neighbors limit dictionary')
            for layer_ind in range(self.dataset.config.num_layers):
                dl = self.dataset.config.first_subsampling_dl * (2**layer_ind)
                if self.dataset.config.deform_layers[layer_ind]:
                    r = dl * self.dataset.config.deform_radius
                else:
                    r = dl * self.dataset.config.conv_radius
                key = '{:.3f}_{:.3f}'.format(dl, r)

                if key in neighb_lim_dict:
                    color = bcolors.OKGREEN
                    v = str(neighb_lim_dict[key])
                else:
                    color = bcolors.FAIL
                    v = '?'
                print('{:}\"{:s}\": {:s}{:}'.format(color, key, v, bcolors.ENDC))

        if redo:
            print("Redo needed, exit!")
            exit(0)


        print('Calibration done in {:.1f}s\n'.format(time.time() - t0))
        return


class ModelNet40CustomBatch:
    """Custom batch definition with memory pinning for ModelNet40"""

    def __init__(self, input_list):
        if len(input_list) == 0:
            return
        # print("Custom batch init")
        # Get rid of batch dimension
        input_list = input_list[0]
        self.input_list = input_list
        # Number of layers
        L = (len(input_list) - 5) // 4
        # print("input_list: ")
        for ind, i in enumerate(input_list):
            # print(input_list[ind].shape)
            if i.shape[0] == 1:
                input_list[ind] = input_list[ind].squeeze(0)
        # print("L: ", L)
        # Extract input tensors from the list of numpy array
        ind = 0
        self.points = [jt.array(nparray) for nparray in input_list[ind:ind+L]]
        # print("points:", self.points[0].shape)
        ind += L
        self.neighbors = [jt.array(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.pools = [jt.array(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.lengths = [jt.array(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.features = jt.array(input_list[ind])
        ind += 1
        self.labels = jt.array(input_list[ind])
        ind += 1
        self.scales = jt.array(input_list[ind])
        ind += 1
        self.rots = jt.array(input_list[ind])
        ind += 1
        self.model_inds = jt.array(input_list[ind])
        # exit(0)
        # for x in self.points:
        #     print(x.dtype)
        # for x in self.neighbors:
        #     print(x.dtype)
        # for x in self.pools:
        #     print(x.dtype)
        # for x in self.lengths:
        #     print(x.dtype)
        # print(self.features.dtype)
        # print(self.labels.dtype)
        # print(self.scales.dtype)
        # print(self.rots.dtype)
        # print(self.model_inds.dtype)
        # exit(0)
        return

    def unstack_points(self, layer=None):
        """Unstack the points"""
        return self.unstack_elements('points', layer)

    def unstack_neighbors(self, layer=None):
        """Unstack the neighbors indices"""
        return self.unstack_elements('neighbors', layer)

    def unstack_pools(self, layer=None):
        """Unstack the pooling indices"""
        return self.unstack_elements('pools', layer)

    def unstack_elements(self, element_name, layer=None, to_numpy=True):
        """
        Return a list of the stacked elements in the batch at a certain layer. If no layer is given, then return all
        layers
        """

        if element_name == 'points':
            elements = self.points
        elif element_name == 'neighbors':
            elements = self.neighbors
        elif element_name == 'pools':
            elements = self.pools[:-1]
        else:
            raise ValueError('Unknown element name: {:s}'.format(element_name))

        all_p_list = []
        for layer_i, layer_elems in enumerate(elements):

            if layer is None or layer == layer_i:

                i0 = 0
                p_list = []
                if element_name == 'pools':
                    lengths = self.lengths[layer_i+1]
                else:
                    lengths = self.lengths[layer_i]

                for b_i, length in enumerate(lengths):

                    elem = layer_elems[i0:i0 + length]
                    if element_name == 'neighbors':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= i0
                    elif element_name == 'pools':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= jt.sum(self.lengths[layer_i][:b_i])
                    i0 += length

                    if to_numpy:
                        p_list.append(elem.numpy())
                    else:
                        p_list.append(elem)

                if layer == layer_i:
                    return p_list

                all_p_list.append(p_list)

        return all_p_list


class Config:
    """
    Class containing the parameters you want to modify for this dataset
    """

    ##################
    # Input parameters
    ##################

    # Dataset name
    dataset = ''

    # Type of network model
    dataset_task = ''

    # Number of classes in the dataset
    num_classes = 0

    # Dimension of input points
    in_points_dim = 3

    # Dimension of input features
    in_features_dim = 1

    # Radius of the input sphere (ignored for models, only used for point clouds)
    in_radius = 1.0

    # Number of CPU threads for the input pipeline
    input_threads = 8

    ##################
    # Model parameters
    ##################

    # Architecture definition. List of blocks
    architecture = []

    # Decide the mode of equivariance and invariance
    equivar_mode = ''
    invar_mode = ''

    # Dimension of the first feature maps
    first_features_dim = 64

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.99

    # For segmentation models : ratio between the segmented area and the input area
    segmentation_ratio = 1.0

    ###################
    # KPConv parameters
    ###################

    # Number of kernel points
    num_kernel_points = 15

    # Size of the first subsampling grid in meter
    first_subsampling_dl = 0.02

    # Radius of convolution in "number grid cell". (2.5 is the standard value)
    conv_radius = 2.5

    # Radius of deformable convolution in "number grid cell". Larger so that deformed kernel can spread out
    deform_radius = 5.0

    # Kernel point influence radius
    KP_extent = 1.0

    # Influence function when d < KP_extent. ('constant', 'linear', 'gaussian') When d > KP_extent, always zero
    KP_influence = 'linear'

    # Aggregation function of KPConv in ('closest', 'sum')
    # Decide if you sum all kernel point influences, or if you only take the influence of the closest KP
    aggregation_mode = 'sum'

    # Fixed points in the kernel : 'none', 'center' or 'verticals'
    fixed_kernel_points = 'center'

    # Use modulateion in deformable convolutions
    modulated = False

    # For SLAM datasets like SemanticKitti number of frames used (minimum one)
    n_frames = 1

    # For SLAM datasets like SemanticKitti max number of point in input cloud + validation
    max_in_points = 0
    val_radius = 51.0
    max_val_points = 50000

    #####################
    # Training parameters
    #####################

    # Network optimizer parameters (learning rate and momentum)
    learning_rate = 1e-3
    momentum = 0.9

    # Learning rate decays. Dictionary of all decay values with their epoch {epoch: decay}.
    lr_decays = {200: 0.2, 300: 0.2}

    # Gradient clipping value (negative means no clipping)
    grad_clip_norm = 100.0

    # Augmentation parameters
    augment_scale_anisotropic = True
    augment_scale_min = 0.9
    augment_scale_max = 1.1
    augment_symmetries = [False, False, False]
    augment_rotation = 'vertical'
    augment_noise = 0.005
    augment_color = 0.7

    # Augment with occlusions (not implemented yet)
    augment_occlusion = 'none'
    augment_occlusion_ratio = 0.2
    augment_occlusion_num = 1

    # Regularization loss importance
    weight_decay = 1e-3

    # The way we balance segmentation loss DEPRECATED
    segloss_balance = 'none'

    # Choose weights for class (used in segmentation loss). Empty list for no weights
    class_w = []

    # Deformable offset loss
    # 'point2point' fitting geometry by penalizing distance from deform point to input points
    # 'point2plane' fitting geometry by penalizing distance from deform point to input point triplet (not implemented)
    deform_fitting_mode = 'point2point'
    deform_fitting_power = 1.0              # Multiplier for the fitting/repulsive loss
    deform_lr_factor = 0.1                  # Multiplier for learning rate applied to the deformations
    repulse_extent = 1.0                    # Distance of repulsion for deformed kernel points

    # Number of batch
    batch_num = 10
    val_batch_num = 10

    # Maximal number of epochs
    max_epoch = 1000

    # Number of steps per epochs
    epoch_steps = 1000

    # Number of validation examples per epoch
    validation_size = 100

    # Number of epoch between each checkpoint
    checkpoint_gap = 50

    # Do we nee to save convergence
    saving = True
    saving_path = None

    def __init__(self):
        """
        Class Initialyser
        """

        # Number of layers
        self.num_layers = len([block for block in self.architecture if 'pool' in block or 'strided' in block]) + 1

        ###################
        # Deform layer list
        ###################
        #
        # List of boolean indicating which layer has a deformable convolution
        #

        layer_blocks = []
        self.deform_layers = []
        arch = self.architecture
        for block_i, block in enumerate(arch):

            # Get all blocks of the layer
            if not ('pool' in block or 'strided' in block or 'global' in block or 'upsample' in block):
                layer_blocks += [block]
                continue

            # Convolution neighbors indices
            # *****************************

            deform_layer = False
            if layer_blocks:
                if np.any(['deformable' in blck for blck in layer_blocks]):
                    deform_layer = True

            if 'pool' in block or 'strided' in block:
                if 'deformable' in block:
                    deform_layer = True

            self.deform_layers += [deform_layer]
            layer_blocks = []

            # Stop when meeting a global pooling or upsampling
            if 'global' in block or 'upsample' in block:
                break

    def load(self, path):

        filename = join(path, 'parameters.txt')
        with open(filename, 'r') as f:
            lines = f.readlines()

        # Class variable dictionary
        for line in lines:
            line_info = line.split()
            if len(line_info) > 2 and line_info[0] != '#':

                if line_info[2] == 'None':
                    setattr(self, line_info[0], None)

                elif line_info[0] == 'lr_decay_epochs':
                    self.lr_decays = {int(b.split(':')[0]): float(b.split(':')[1]) for b in line_info[2:]}

                elif line_info[0] == 'architecture':
                    self.architecture = [b for b in line_info[2:]]

                elif line_info[0] == 'augment_symmetries':
                    self.augment_symmetries = [bool(int(b)) for b in line_info[2:]]

                elif line_info[0] == 'num_classes':
                    if len(line_info) > 3:
                        self.num_classes = [int(c) for c in line_info[2:]]
                    else:
                        self.num_classes = int(line_info[2])

                elif line_info[0] == 'class_w':
                    self.class_w = [float(w) for w in line_info[2:]]

                elif hasattr(self, line_info[0]):
                    attr_type = type(getattr(self, line_info[0]))
                    if attr_type == bool:
                        setattr(self, line_info[0], attr_type(int(line_info[2])))
                    else:
                        setattr(self, line_info[0], attr_type(line_info[2]))

        self.saving = True
        self.saving_path = path
        self.__init__()

    def save(self):

        with open(join(self.saving_path, 'parameters.txt'), "w") as text_file:

            text_file.write('# -----------------------------------#\n')
            text_file.write('# Parameters of the training session #\n')
            text_file.write('# -----------------------------------#\n\n')

            # Input parameters
            text_file.write('# Input parameters\n')
            text_file.write('# ****************\n\n')
            text_file.write('dataset = {:s}\n'.format(self.dataset))
            text_file.write('dataset_task = {:s}\n'.format(self.dataset_task))
            if type(self.num_classes) is list:
                text_file.write('num_classes =')
                for n in self.num_classes:
                    text_file.write(' {:d}'.format(n))
                text_file.write('\n')
            else:
                text_file.write('num_classes = {:d}\n'.format(self.num_classes))
            text_file.write('in_points_dim = {:d}\n'.format(self.in_points_dim))
            text_file.write('in_features_dim = {:d}\n'.format(self.in_features_dim))
            text_file.write('in_radius = {:.6f}\n'.format(self.in_radius))
            text_file.write('input_threads = {:d}\n\n'.format(self.input_threads))

            # Model parameters
            text_file.write('# Model parameters\n')
            text_file.write('# ****************\n\n')

            text_file.write('architecture =')
            for a in self.architecture:
                text_file.write(' {:s}'.format(a))
            text_file.write('\n')
            text_file.write('equivar_mode = {:s}\n'.format(self.equivar_mode))
            text_file.write('invar_mode = {:s}\n'.format(self.invar_mode))
            text_file.write('num_layers = {:d}\n'.format(self.num_layers))
            text_file.write('first_features_dim = {:d}\n'.format(self.first_features_dim))
            text_file.write('use_batch_norm = {:d}\n'.format(int(self.use_batch_norm)))
            text_file.write('batch_norm_momentum = {:.6f}\n\n'.format(self.batch_norm_momentum))
            text_file.write('segmentation_ratio = {:.6f}\n\n'.format(self.segmentation_ratio))

            # KPConv parameters
            text_file.write('# KPConv parameters\n')
            text_file.write('# *****************\n\n')

            text_file.write('first_subsampling_dl = {:.6f}\n'.format(self.first_subsampling_dl))
            text_file.write('num_kernel_points = {:d}\n'.format(self.num_kernel_points))
            text_file.write('conv_radius = {:.6f}\n'.format(self.conv_radius))
            text_file.write('deform_radius = {:.6f}\n'.format(self.deform_radius))
            text_file.write('fixed_kernel_points = {:s}\n'.format(self.fixed_kernel_points))
            text_file.write('KP_extent = {:.6f}\n'.format(self.KP_extent))
            text_file.write('KP_influence = {:s}\n'.format(self.KP_influence))
            text_file.write('aggregation_mode = {:s}\n'.format(self.aggregation_mode))
            text_file.write('modulated = {:d}\n'.format(int(self.modulated)))
            text_file.write('n_frames = {:d}\n'.format(self.n_frames))
            text_file.write('max_in_points = {:d}\n\n'.format(self.max_in_points))
            text_file.write('max_val_points = {:d}\n\n'.format(self.max_val_points))
            text_file.write('val_radius = {:.6f}\n\n'.format(self.val_radius))

            # Training parameters
            text_file.write('# Training parameters\n')
            text_file.write('# *******************\n\n')

            text_file.write('learning_rate = {:f}\n'.format(self.learning_rate))
            text_file.write('momentum = {:f}\n'.format(self.momentum))
            text_file.write('lr_decay_epochs =')
            for e, d in self.lr_decays.items():
                text_file.write(' {:d}:{:f}'.format(e, d))
            text_file.write('\n')
            text_file.write('grad_clip_norm = {:f}\n\n'.format(self.grad_clip_norm))


            text_file.write('augment_symmetries =')
            for a in self.augment_symmetries:
                text_file.write(' {:d}'.format(int(a)))
            text_file.write('\n')
            text_file.write('augment_rotation = {:s}\n'.format(self.augment_rotation))
            text_file.write('augment_noise = {:f}\n'.format(self.augment_noise))
            text_file.write('augment_occlusion = {:s}\n'.format(self.augment_occlusion))
            text_file.write('augment_occlusion_ratio = {:.6f}\n'.format(self.augment_occlusion_ratio))
            text_file.write('augment_occlusion_num = {:d}\n'.format(self.augment_occlusion_num))
            text_file.write('augment_scale_anisotropic = {:d}\n'.format(int(self.augment_scale_anisotropic)))
            text_file.write('augment_scale_min = {:.6f}\n'.format(self.augment_scale_min))
            text_file.write('augment_scale_max = {:.6f}\n'.format(self.augment_scale_max))
            text_file.write('augment_color = {:.6f}\n\n'.format(self.augment_color))

            text_file.write('weight_decay = {:f}\n'.format(self.weight_decay))
            text_file.write('segloss_balance = {:s}\n'.format(self.segloss_balance))
            text_file.write('class_w =')
            for a in self.class_w:
                text_file.write(' {:.6f}'.format(a))
            text_file.write('\n')
            text_file.write('deform_fitting_mode = {:s}\n'.format(self.deform_fitting_mode))
            text_file.write('deform_fitting_power = {:.6f}\n'.format(self.deform_fitting_power))
            text_file.write('deform_lr_factor = {:.6f}\n'.format(self.deform_lr_factor))
            text_file.write('repulse_extent = {:.6f}\n'.format(self.repulse_extent))
            text_file.write('batch_num = {:d}\n'.format(self.batch_num))
            text_file.write('val_batch_num = {:d}\n'.format(self.val_batch_num))
            text_file.write('max_epoch = {:d}\n'.format(self.max_epoch))
            if self.epoch_steps is None:
                text_file.write('epoch_steps = None\n')
            else:
                text_file.write('epoch_steps = {:d}\n'.format(self.epoch_steps))
            text_file.write('validation_size = {:d}\n'.format(self.validation_size))
            text_file.write('checkpoint_gap = {:d}\n'.format(self.checkpoint_gap))

class Modelnet40Config(Config):
    """
    Override the parameters you want to modify for this dataset
    """

    ####################
    # Dataset parameters
    ####################

    # Dataset name
    dataset = 'ModelNet40'

    # Number of classes in the dataset (This value is overwritten by dataset class when Initializating dataset).
    num_classes = 40

    # Type of task performed on this dataset (also overwritten)
    dataset_task = 'classification'

    # Number of CPU threads for the input pipeline
    input_threads = 10

    #########################
    # Architecture definition
    #########################

    # Define layers
    architecture = ['simple',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'global_average']

    ###################
    # KPConv parameters
    ###################

    # Number of kernel points
    num_kernel_points = 15

    # Size of the first subsampling grid in meter
    first_subsampling_dl = 0.02

    # Radius of convolution in "number grid cell". (2.5 is the standard value)
    conv_radius = 2.5

    # Radius of deformable convolution in "number grid cell". Larger so that deformed kernel can spread out
    deform_radius = 6.0

    # Radius of the area of influence of each kernel point in "number grid cell". (1.0 is the standard value)
    KP_extent = 1.2

    # Behavior of convolutions in ('constant', 'linear', 'gaussian')
    KP_influence = 'linear'

    # Aggregation function of KPConv in ('closest', 'sum')
    aggregation_mode = 'sum'

    # Choice of input features
    in_features_dim = 1

    # Can the network learn modulations
    modulated = True

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.05

    # Deformable offset loss
    # 'point2point' fitting geometry by penalizing distance from deform point to input points
    # 'point2plane' fitting geometry by penalizing distance from deform point to input point triplet (not implemented)
    deform_fitting_mode = 'point2point'
    deform_fitting_power = 1.0              # Multiplier for the fitting/repulsive loss
    deform_lr_factor = 0.1                  # Multiplier for learning rate applied to the deformations
    repulse_extent = 1.2                    # Distance of repulsion for deformed kernel points

    #####################
    # Training parameters
    #####################

    # Maximal number of epochs
    max_epoch = 500

    # Learning rate management
    learning_rate = 1e-2
    momentum = 0.98
    lr_decays = {i: 0.1**(1/100) for i in range(1, max_epoch)}
    grad_clip_norm = 100.0

    # Number of batch
    batch_num = 10

    # Number of steps per epochs
    epoch_steps = 300

    # Number of validation examples per epoch
    validation_size = 30

    # Number of epoch between each checkpoint
    checkpoint_gap = 50

    # Augmentations
    augment_scale_anisotropic = True
    augment_symmetries = [True, True, True]
    augment_rotation = 'none'
    augment_scale_min = 0.8
    augment_scale_max = 1.2
    augment_noise = 0.001
    augment_color = 1.0

    # The way we balance segmentation loss
    #   > 'none': Each point in the whole batch has the same contribution.
    #   > 'class': Each class has the same contribution (points are weighted according to class balance)
    #   > 'batch': Each cloud in the batch has the same contribution (points are weighted according cloud sizes)
    segloss_balance = 'none'

    # Do we nee to save convergence
    saving = True
    saving_path = None


if __name__ == "__main__":
    
    cfg = Modelnet40Config()
    training_dataset = ModelNet40Dataset(cfg, train=True)
    training_sampler = ModelNet40Sampler(training_dataset, balance_labels=True)
    training_sampler.calibration()
    # print("### original data ###")
    # for data in training_dataset:
    #     for x in data:
    #         print(x.shape)
    #     break
    print("### sampled data ###")
    sampled_idx = next(training_sampler.__iter__())
    sampled_data = training_dataset.__getitem__(sampled_idx)
    # print(sampled_data)
    for x in sampled_data:
        print(x.shape)
    # sampled_idx = next(training_sampler.__iter__())
    # sampled_data = training_dataset.__getitem__(sampled_idx)
    # # print(sampled_data)
    # for x in sampled_data:
    #     print(x.shape)
    # break