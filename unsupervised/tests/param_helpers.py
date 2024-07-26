import numpy as np
import parametrize_from_file as pff

with_py = pff.Namespace()
with_math = pff.Namespace('from math import *')
with_np = pff.Namespace('import numpy as np')

def image(params):
    shape = tuple(int(x) for x in params['shape'].split())
    img = np.zeros(shape)

    for k, v in params.get('voxels', {}).items():
        i = tuple(int(x) for x in k.split())
        img[i] = float(v)

    return img



