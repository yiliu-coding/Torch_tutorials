import os
from os.path import dirname, abspath


def get_path_file(*args):
    d = dirname(dirname(dirname(abspath(__file__))))

    if any(args):
        print(args)
        file_name = args[0]
        image_file = os.path.join(d, 'data', file_name)
    else:
        image_file = os.path.join(d, 'data', 'image_cluster_sample.jpg')

    print("file_path: ", image_file)

    return os.path.join(d, 'data'), image_file

print(get_path_file('aa.file'))