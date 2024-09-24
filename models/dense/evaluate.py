import tensorflow as tf
import sympy as sp
import numpy as np
import sys

sys.path.append('.')
from custom_functions import *

#Converts sympy monomial into its vector representation
def str2vector(str):
    if str == "w":
        return np.array([0,1,0,0,0,0,0,0,0])
    elif str == "1/w" or str == "w^-1" or str == "w**-1":
        return np.array([0,0,1,0,0,0,0,0,0])
    elif str == "z":
        return np.array([0,0,0,1,0,0,0,0,0])
    elif str == "1/z" or str == "z^-1" or str == "z**-1":
        return np.array([0,0,0,0,1,0,0,0,0])
    elif str == "zw" or str == "wz" or str == "z*w" or str == "w*z":
        return np.array([0,0,0,0,0,1,0,0,0])
    elif str == "z/w" or str == "z*1/w" or str == "1/w*z" or str == "1/wz" or str == "zw^-1" or str == "w^-1z" or str == "zw**-1" or str == "w**-1z" or  str == "z*w^-1" or str == "w^-1*z" or str == "z*w**-1" or str == "w**-1*z":
        return np.array([0,0,0,0,0,0,1,0,0])
    elif str == "w/z" or str == "w*1/z" or str == "1/z*w" or str == "1/zw" or str == "wz^-1" or str == "z^-1w" or str == "wz**-1" or str == "z**-1w" or str == "w*z^-1" or str == "z^-1*w" or str == "w*z**-1" or str == "z**-1*w":
        return np.array([0,0,0,0,0,0,0,1,0])
    elif str == "1/(wz)" or str == "1/(w*z)" or str == "1/(zw)" or str == "1/(z*w)" or str == "1/z*1/w" or str == "1/w/z" or str == "1/z/w" or str == "w^-1z^-1" or str == "z^-1w^-1" or str == "w^-1*z^-1" or str == "z^-1*w^-1" or str == "w**-1z**-1" or str == "z**-1w**-1" or str == "w**-1*z**-1" or str == "z**-1*w**-1":
        return np.array([0,0,0,0,0,0,0,0,1])
    else:
        return np.array([999,999,999,999,999,999,999,999,999])

#Converts sympy polynomial into its vector representation
def sympyExp2vector(exp):
    if str(exp) == "0":
        return np.array([0,0,0,0,0,0,0,0,0])
    elif "w" not in str(exp) and "z" not in str(exp):
        return int(exp)*np.array([1,0,0,0,0,0,0,0,0])
    else:
        monomials = exp.as_ordered_terms()
        vector = np.array([0,0,0,0,0,0,0,0,0])
        for term in monomials:
            term_str = str(term)
            cst = 1
            if "w" not in term_str and "z" not in term_str:
                cst = int(term_str)
                term_vector = np.array([1,0,0,0,0,0,0,0,0])
            else:
                if "w" not in term_str:
                    temp_index = term_str.index("z")
                elif "z" not in term_str:
                    temp_index = term_str.index("w")
                else:
                    temp_index = min(term_str.index("w"),term_str.index("z"))
                
                if temp_index == 0:
                    cst = 1
                elif temp_index == 1 and term_str[0] == "-":
                    cst = -1
                elif term_str[temp_index-1] == "/":
                    cst = int(term_str[:temp_index-1])
                    term_str = "1" + term_str[temp_index-1:]
                    temp_index = 0
                elif term_str[temp_index-2] == "/":
                    cst = int(term_str[:temp_index-2])
                    term_str = "1" + term_str[temp_index-2:]
                    temp_index = 0
                else:
                    cst = int(term_str[:temp_index-1])
                
                term_vector = str2vector(term_str[temp_index:])

            vector += cst*term_vector
        return vector

#Converts sympy matrix to its tensor representation
def kasteleynMat2Vector(mat):
    shape = (mat.shape[0], mat.shape[1], 9)
    vect_mat = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            vect_mat[i][j] = sympyExp2vector(mat[i,j])

    return vect_mat

if __name__ == "__main__":

    # Input constants used to identify correct dataset
    MATRIX_SIZE = 128
    ITERATIONS = 1000

    # Network hyperparameters
    HP_DENSE_SHAPE = 64
    HP_BATCH_SIZE = 64
    HP_ACTIVATION = 'relu'

    METRIC_ACCURACY = 'mae'

    w = sp.Symbol('w')
    z = sp.Symbol('z')

	# ----- a few examples of Kastelyn matrices -----
    # [4,2]
    sympyMat1 = sp.Matrix(
        [[-1, 1, 0, 0, 0, 0, 1, -1],
         [z, 1, 0, 0, 0, 0, z, 1],
         [1, 1, -1, 1, 0, 0, 0, 0],
         [-z, 1, z, -1, 0, 0, 0, 0],
         [0, 0, -1, 1, -1, 1, 0, 0],
         [0, 0, z, -1, z, -1, 0, 0],
         [0, 0, 0, 0, -w, w, 1, 1],
         [0, 0, 0, 0, w*z, -w, z, 1]
        ])
    # same but rows flipped
    sympyMat2 = sp.Matrix(
        [[-1, 1, 0, 0, 0, 0, 1, -1],
         [0, 0, z, -1, z, -1, 0, 0],
         [z, 1, 0, 0, 0, 0, z, 1],
         [1, 1, -1, 1, 0, 0, 0, 0],
         [-z, 1, z, -1, 0, 0, 0, 0],
         [0, 0, -1, 1, -1, 1, 0, 0],
         [0, 0, 0, 0, -w, w, 1, 1],
         [0, 0, 0, 0, w*z, -w, z, 1]
        ])
    # same but w <-> z
    sympyMat3 = sp.Matrix(
        [[-1, 1, 0, 0, 0, 0, 1, -1],
         [w, 1, 0, 0, 0, 0, w, 1],
         [1, 1, -1, 1, 0, 0, 0, 0],
         [-w, 1, w, -1, 0, 0, 0, 0],
         [0, 0, -1, 1, -1, 1, 0, 0],
         [0, 0, w, -1, w, -1, 0, 0],
         [0, 0, 0, 0, -z, z, 1, 1],
         [0, 0, 0, 0, w*z, -z, w, 1]
        ])
    # same but row flipped
    sympyMat4 = sp.Matrix(
        [[-1, 1, 0, 0, 0, 0, 1, -1],
         [w, 1, 0, 0, 0, 0, w, 1],
         [1, 1, -1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, -z, z, 1, 1],
         [-w, 1, w, -1, 0, 0, 0, 0],
         [0, 0, -1, 1, -1, 1, 0, 0],
         [0, 0, w, -1, w, -1, 0, 0],
         [0, 0, 0, 0, w*z, -z, w, 1]
        ])
    # same but 2 row flipped and w -> 1/w
    sympyMat5 = sp.Matrix(
        [[-1/w, 1, 1/w, -1, 0, 0, 0, 0],
         [0, 0, -1, 1, -1, 1, 0, 0],
         [-1, 1, 0, 0, 0, 0, 1, -1],
         [1/w, 1, 0, 0, 0, 0, 1/w, 1],
         [1, 1, -1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, -z, z, 1, 1],
         [0, 0, 1/w, -1, 1/w, -1, 0, 0],
         [0, 0, 0, 0, z/w, -z, 1/w, 1]
        ])
    # [2,6]
    sympyMat6 = sp.Matrix([
			[-1,0,0,0,0,1,1,0,0,0,0,-1],
			[1,1,0,0,0,0,1,-1,0,0,0,0],
			[0,1,1,0,0,0,0,1,-1,0,0,0],
			[0,0,1,1,0,0,0,0,1,-1,0,0],
			[0,0,0,1,1,0,0,0,0,1,-1,0],
			[0,0,0,0,z,1,0,0,0,0,z,1],
			[w,0,0,0,0,w,1,0,0,0,0,1],
			[w,w,0,0,0,0,1,1,0,0,0,0],
			[0,w,w,0,0,0,0,1,1,0,0,0],
			[0,0,w,w,0,0,0,0,1,1,0,0],
			[0,0,0,w,w,0,0,0,0,1,1,0],
			[0,0,0,0,w*z,w,0,0,0,0,z,-1]
        ])
    # Same but w <-> z
    sympyMat7 = sp.Matrix([
			[-1,0,0,0,0,1,1,0,0,0,0,-1],
			[1,1,0,0,0,0,1,-1,0,0,0,0],
			[0,1,1,0,0,0,0,1,-1,0,0,0],
			[0,0,1,1,0,0,0,0,1,-1,0,0],
			[0,0,0,1,1,0,0,0,0,1,-1,0],
			[0,0,0,0,w,1,0,0,0,0,w,1],
			[z,0,0,0,0,z,1,0,0,0,0,1],
			[z,z,0,0,0,0,1,1,0,0,0,0],
			[0,z,z,0,0,0,0,1,1,0,0,0],
			[0,0,z,z,0,0,0,0,1,1,0,0],
			[0,0,0,z,z,0,0,0,0,1,1,0],
			[0,0,0,0,w*z,z,0,0,0,0,w,-1]
        ])
    # [2,2]
    sympyMat8 = sp.Matrix([
        [-1,1,1,-1],
        [z,1,z,1],
		[w,w,1,1],
		[-w*z,w,z,-1]])
	# -----------------------------------------------
    
	# ----- convert matrices to tensorflow tensor -----
    example_matrices = [sympyMat1, sympyMat2, sympyMat3, sympyMat4, sympyMat5, sympyMat6, sympyMat7, sympyMat8]
    input_matrices = [tf.sparse.to_dense(tf.sparse.expand_dims(tf.sparse.reset_shape(tf.sparse.from_dense(np.array(kasteleynMat2Vector(mat))), new_shape=(MATRIX_SIZE, MATRIX_SIZE, 9)), axis=0)) for mat in example_matrices]

    input_data = tf.concat(input_matrices, axis=0)
    print(input_data.shape)
    # --------------------------------------------------
    
	# Define model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(MATRIX_SIZE,MATRIX_SIZE,9)),
        RandomFlipLayer(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(HP_DENSE_SHAPE, activation=HP_ACTIVATION),
        tf.keras.layers.Dense(2)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=[METRIC_ACCURACY])
    model.build()
    model.summary()

    logdir = 'ms' + str(MATRIX_SIZE) + '_b' + str(HP_BATCH_SIZE) + '_ds' + str(HP_DENSE_SHAPE) + '_ac' + HP_ACTIVATION

	# Predict output
    latest_epoch = checkpoint_loader(path='models/dense/checkpoints/'+logdir, model=model)
    predictions = model.predict(input_data)
    print(predictions)