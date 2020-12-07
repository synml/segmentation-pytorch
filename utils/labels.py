from collections import namedtuple

VOCClasses = namedtuple('VOCClasses', ['name', 'classId', 'color'])
classes = [
    VOCClasses('_background', 0, (0, 0, 0)),
    VOCClasses('aeroplane', 1, (128, 0, 0)),
    VOCClasses('bicycle', 2, (0, 128, 0)),
    VOCClasses('bird', 3, (128, 128, 0)),
    VOCClasses('boat', 4, (0, 0, 128)),
    VOCClasses('bottle', 5, (128, 0, 128)),
    VOCClasses('bus', 6, (0, 128, 128)),
    VOCClasses('car', 7, (128, 128, 128)),
    VOCClasses('cat', 8, (64, 0, 0)),
    VOCClasses('chair', 9, (192, 0, 0)),
    VOCClasses('cow', 10, (64, 128, 0)),
    VOCClasses('diningtable', 11, (192, 128, 0)),
    VOCClasses('dog', 12, (64, 0, 128)),
    VOCClasses('horse', 13, (192, 0, 128)),
    VOCClasses('motorbike', 14, (64, 128, 128)),
    VOCClasses('person', 15, (192, 128, 128)),
    VOCClasses('pottedplant', 16, (0, 64, 0)),
    VOCClasses('sheep', 17, (128, 64, 0)),
    VOCClasses('sofa', 18, (0, 192, 0)),
    VOCClasses('train', 19, (128, 192, 0)),
    VOCClasses('tvmonitor', 20, (0, 64, 128)),
]
