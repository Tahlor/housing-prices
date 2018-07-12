def reset_seed(SEED=42):
    import numpy as np
    import tensorflow as tf
    import random as rn
    import os
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(SEED)
    rn.seed(SEED)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    from keras import backend as K
    tf.set_random_seed(SEED)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)