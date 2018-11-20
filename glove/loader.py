import numpy as np
import pyprog
import subprocess
import os

CUR_FILE_PATH = os.path.dirname(os.path.realpath(__file__))
LOADER_LENGTH = 30
GLOVE_EMBED_DIR = '{}/glove'.format(CUR_FILE_PATH)

def load(d=50):
    # assemble filename
    fname = '{}/glove.6B.{}d.txt'.format(GLOVE_EMBED_DIR, d)
    if not os.path.isfile(fname):
        raise Exception('file {} does not exist'.format(fname))

    # get file length
    fl = int(
        subprocess.run(
            ['wc', '-l' ,fname],
            stdout=subprocess.PIPE
        ).stdout.strip().split()[0]
    )

    print('loading glove embeddings from {}'.format(fname))

    # setup
    prog = pyprog.ProgressBar("", "")
    out = {}
    f = open(fname, 'r')
    c = 0

    # read line by line
    # coutesy of https://stackoverflow.com/questions/37793118/load-pretrained-glove-vectors-in-python
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        out[word] = embedding
        c += 1
        prog.set_stat(int(c / fl * 100))
        prog.update()

    prog.end()
    print('done!')

    return out




