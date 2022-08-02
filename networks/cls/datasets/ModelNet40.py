import jittor as jt

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
        # for i in input_list:
        #     print(i.shape)
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
