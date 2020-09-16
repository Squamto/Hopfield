import numpy
from math import log
import struct
import random


class MNISTHandler:
    file_names = {'images': "./util/mnist/t10k-images.idx3-ubyte", 'labels': "./util/mnist/t10k-labels.idx1-ubyte"}

    def __init__(self):
        self.files = {}
        for key, value in self.file_names.items():
            self.files[key] = open(value, 'rb')

        # inti images-file
        self.files["images"].seek(0)
        _ = struct.unpack('>4B', self.files["images"].read(4))
        self.max_images = self.remaining_images = struct.unpack('>I', self.files["images"].read(4))[0]
        self.number_rows_images = struct.unpack('>I', self.files["images"].read(4))[0]
        self.number_columns_images = struct.unpack('>I', self.files["images"].read(4))[0]

        # init labels-file
        self.files["labels"].seek(0)
        _ = struct.unpack('>4B', self.files["labels"].read(4))
        self.max_labels = struct.unpack(">I", self.files["labels"].read(4))[0]

    def __del__(self):
        for f in self.files.values():
            f.close()

    def get_pictures(self, count=1):
        if count > self.max_images:
            print("".join(["Only ", str(self.max_images), " availible"]))
            count = self.max_images
        self.remaining_images -= count
        if self.remaining_images < 0:
            self.files["images"].seek(16)
            self.files["labels"].seek(8)
            self.remaining_images = self.max_images
        bytes_total = count * self.number_rows_images * self.number_columns_images
        images_array = numpy.asarray(struct.unpack('>' + 'B' * bytes_total,
                                                   self.files["images"].read(bytes_total))).reshape(
            (count, self.number_rows_images, self.number_columns_images))

        labels_array = numpy.asarray(struct.unpack('>' + 'B' * count,
                                                   self.files["labels"].read(count)))

        return images_array, labels_array

    def get_pictures_by_label(self, count_dict):
        pictures_array = numpy.empty(
            (10, max(count_dict.values()), self.number_rows_images, self.number_columns_images))
        for x in range(10):
            if x not in count_dict:
                count_dict[x] = 0

        finished = False
        while not finished:
            next_image, next_label = self.get_pictures(1)
            if count_dict[next_label[0]] > 0:
                count_dict[next_label[0]] -= 1
                pictures_array[next_label[0], count_dict[next_label[0]]] = next_image[0]
            if sum(count_dict.values()) == 0:
                finished = True

        return pictures_array


class NewWeightContainer(numpy.ndarray):
    def __new__(cls, size, random=False, ex=False):
        arr = numpy.empty((size, size), dtype=float)
        if random:
            if ex:
                arr = numpy.random.random((size, size))
            else:
                for x in range(size):
                    arr[numpy.arange(x), x] = numpy.random.random(x) - 0.5

        arr = arr.view(cls)
        arr.symetric = True
        arr.zero_diagonal = True
        return arr

    def __array_finalize__(self, obj):
        if obj is None: return

        self.symetric = getattr(obj, 'symetric', True)
        self.zero_diagonal = getattr(obj, 'zero_diagonal', True)

    def __setitem__(self, key, value):
        if type(key) == tuple and self.symetric and [int for _ in key] == [type(x) for x in key] and len(key) == 2:
            super(NewWeightContainer, self).__setitem__((max(key), min(key)), value)
        else:
            super(NewWeightContainer, self).__setitem__(key, value)

    def __getitem__(self, item):
        if type(item) == tuple and len(item) == 2 and [int for _ in item] == [type(x) for x in item]:
            if self.zero_diagonal and item[0] == item[1]:
                return 0
            else:
                if self.symetric:
                    return super(NewWeightContainer, self).__getitem__((max(item), min(item)))
                else:
                    return super(NewWeightContainer, self).__getitem__(item)
        else:
            return super(NewWeightContainer, self).__getitem__(item)

    def get_weights_for_point(self, index):
        if self.symetric:
            a = self[numpy.arange(index), index]
            b = self[index, index:]
            arr = numpy.append(a, b)
            if self.zero_diagonal:
                arr[index] = 0
        else:
            arr = self[index]
        return arr

    def get_formatted(self):
        format_string = ""
        number_format = ""
        if self.dtype == int:
            format_string = ''.join(('0', str(int(log(self.max(), 10)) + 1), 'd'))
            number_format = ''.join(('0', str(int(log(self.max(), 10)) + 1), 'd'))
        elif self.dtype == float:
            format_string = '=+5.4f'
            number_format = '^7d'
        else:
            print(self.dtype)

        s = "\n\n"
        s = ''.join([s, format(0, number_format), '  |'])
        s += "|".join([format(x, number_format) for x in range(self.shape[0])])
        s += "|\n\n"
        for y in range(self.shape[0]):
            s = ''.join([s, format(y, number_format), '  '])
            for x in range(self.shape[0]):
                s = "|".join((s, format(self[x, y], format_string)))
            s += '|\n'
        return s

    def clear(self):
        self[:, :] = numpy.zeros(self.shape)


class HopfieldNetwork:

    def __init__(self, network_size=(28, 28), random=False):

        self.network_size = self.w, self.h = network_size
        self.order = 0
        self.sync = 0
        self.step_size = 1
        self.possible_values = [-1, 1]
        self.values = numpy.ones((self.w, self.h)) * -1
        self.old_values = numpy.empty(self.values.shape)
        self.weights = NewWeightContainer(self.w * self.h, random)
        self.saved_states = numpy.array([])
        self.current_x = 0
        self.current_y = 0
        self.remaining_indizes = []

    def randomise_values(self):
        print("randomising Values")
        self.values = numpy.random.choice([-1, 1], self.network_size)

    def clear_values(self):
        print("clearing Values")
        self.values = numpy.ones(self.network_size) * -1
        self.remaining_indizes = []

    def noise_values(self, p=0.3):
        print("noising Values")
        mask = noise = numpy.random.choice([0, 1], self.values.shape, p=[1 - p, p]) == 1
        noise = numpy.random.choice(self.possible_values, self.values.shape)
        self.values[mask] = noise[mask]

    def randomise_weights(self):
        print("randomising Weights")
        sym = self.weights.symetric
        diag = self.weights.zero_diagonal
        self.clear_weights_and_savedstates()
        self.weights = NewWeightContainer(self.w * self.h, True)
        self.weights.symetric = sym
        self.weights.zero_diagonal = diag

    def clear_weights_and_savedstates(self):
        print("clearing weights")
        self.weights.clear()
        self.weights = NewWeightContainer(self.w * self.h)
        self.saved_states = numpy.array([])

    def save_state(self):
        print("saving state")
        if self.saved_states.size == 0:
            self.saved_states = numpy.array([self.values])
        else:
            self.saved_states = numpy.append(self.saved_states, [self.values], axis=0)

    def remove_saved_state(self, idx):
        if self.saved_states.shape[0] == 1:
            self.saved_states = numpy.array([])
        else:
            self.saved_states = numpy.delete(self.saved_states, idx, axis=0)
        self.calculate_weights()

    def calculate_weights(self):
        print("calculating weights")
        self.weights.clear()
        for state in self.saved_states:
            self.weights += numpy.outer(state.ravel(), state.ravel())
        # numpy.fill_diagonal(self.weights, 0)

    def set_sync(self, mode):
        if mode:
            self.sync = 1
            self.current_x = 0
            self.current_y = 0
            self.old_values = self.values.copy()
        else:
            self.old_values = numpy.empty(self.values.shape)
            self.sync = 0

    def step(self):
        try:
            if self.order == 0:
                self.current_x += 1
                self.current_x = self.current_x % self.w
                if self.current_x == 0:
                    self.current_y += 1
                    self.current_y = self.current_y % self.h

            elif self.order == 1:
                self.current_x = numpy.random.randint(low=0, high=self.w)
                self.current_y = numpy.random.randint(low=0, high=self.h)

            elif self.order == 2:
                if len(self.remaining_indizes) == 0:
                    for y in range(self.h):
                        for x in range(self.w):
                            self.remaining_indizes.append((x, y))

                i, (self.current_x, self.current_y) = random.choice(list(enumerate(self.remaining_indizes)))
                self.remaining_indizes.pop(i)

            self.values[self.current_y, self.current_x] = 1 if self.get_np_sum(self.current_x,
                                                                               self.current_y) >= 0 else -1
            
        except Exception as e:
            print(e)

    def sync_step(self):            
        self.values[self.current_y, self.current_x] = 1 if self.get_np_sum(self.current_x,
                                                                               self.current_y) >= 0 else -1
        self.current_x += 1
        self.current_x = self.current_x % self.w
        if self.current_x == 0:
            self.current_y += 1
            self.current_y = self.current_y % self.h
        

        if self.current_x == 0 and self.current_y == 0:
           return
        else:
            self.sync_step() 

    def iterate(self):
        if self.sync == 1:
            self.old_values = self.values.copy()
            self.current_x = 0
            self.current_y = 0
            self.sync_step()
        else:
            for i in range(int(self.step_size)):
                self.step()
            

    def get_np_sum(self, x, y):
        ws = self.weights.get_weights_for_point(x + y * self.w)
        vs = self.values.ravel() if self.sync == 0 else self.old_values.ravel()
        return (ws * vs).sum()

    def load_image(self, image):
        if self.values.shape != image.shape:
            print("Wrong Shape")
        else:
            self.values[image > 180] = 1
            self.values[image <= 180] = -1

'''
def draw_array_to_surface(array, surface, background_color=(100, 100, 100), show_marker=False, current_pos=(-1, -1)):
    width, height = surface.get_size()
    w, h = array.shape
    surface.fill(background_color)
    if w <= width and h <= height:
        rect_size = int(min(width // w, height // h))
        y0 = int((height - rect_size * h) / 2)
        x0 = int((width - rect_size * w) / 2)
        for y in range(h):
            for x in range(w):
                rect = pygame.Rect((x0 + x * rect_size, y0 + y * rect_size), (rect_size, rect_size))
                color = (0, 0, 0) if array[y, x] == -1 else (255, 255, 255)
                if show_marker and x == current_pos[0] and y == current_pos[1]:
                    color = (255, 0, 0)
                pygame.draw.rect(surface, color, rect)
    else:
        print("Surface to small")
'''

def load_example(network, idx):
    if idx == 0:
        network.clear_values()
        network.values[network.h // 2:] *= -1
        network.weights.clear()
        network.weights.symetric = True
        half = network.h // 2 * network.w
        network.weights[:half, half:] = 1
        network.weights[half:, :half] = 1
        network.set_sync(True)
    elif idx == 1:
        network.weights = (numpy.random.random(network.weights.shape) - 0.5).view(NewWeightContainer)
        network.weights.symetric = False
        network.set_sync(False)
    elif idx == 2:
        network.weights = (numpy.eye(network.weights.shape[0]) * -1).view(NewWeightContainer)
        numpy.fill_diagonal(network.weights, -1)
        network.weights.zero_diagonal = False
        network.set_sync(False)


if __name__ == "__main__":
    pass