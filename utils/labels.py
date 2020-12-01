from collections import namedtuple

Label = namedtuple('Label', ['name', 'classId', 'color'])
labels = [
    Label('_background',    0, (0, 0, 0)),
    Label('aeroplane',      1, (128, 0, 0)),
    Label('bicycle',        2, (0, 128, 0)),
    Label('bird',           3, (128, 128, 0)),
    Label('boat',           4, (0, 0, 128)),
    Label('bottle',         5, (128, 0, 128)),
    Label('bus',            6, (0, 128, 128)),
    Label('car',            7, (128, 128, 128)),
    Label('cat',            8, (64, 0, 0)),
    Label('chair',          9, (192, 0, 0)),
    Label('cow',            10, (64, 128, 0)),
    Label('diningtable',    11, (192, 128, 0)),
    Label('dog',            12, (64, 0, 128)),
    Label('horse',          13, (192, 0, 128)),
    Label('motorbike',      14, (64, 128, 128)),
    Label('person',         15, (192, 128, 128)),
    Label('pottedplant',    16, (0, 64, 0)),
    Label('sheep',          17, (128, 64, 0)),
    Label('sofa',           18, (0, 192, 0)),
    Label('train',          19, (128, 192, 0)),
    Label('tvmonitor',      20, (0, 64, 128)),
]
