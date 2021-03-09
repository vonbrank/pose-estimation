import pickle


def load_file(filename):
    with open(filename, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    return data


data = load_file("cifar-10-batches-py/data_batch_1")
print(data['data'])
print(data['data'].shape)

