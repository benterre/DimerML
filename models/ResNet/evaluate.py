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
    MATRIX_SIZE = 18 # 18 for unbalanced y60, 22 for y40, 36 for balanced y60
    NUM_CLASSES = 16 # 5 for y40, 16 for y60, 4 for balanced y60

    # Network hyperparameters
    HP_BATCH_SIZE = 128
    HP_ACTIVATION = 'relu'

    METRIC_ACCURACY = 'accuracy'

    w = sp.Symbol('w')
    z = sp.Symbol('z')

	# ----- a few examples of Kastelyn matrices -----
    # original y60 matrix
    sympyMat1 = sp.Matrix(
        [[-1-z,1,0,0,0,w],
         [1,1+1/z,1,0,0,0],
         [0,1,-1-z,1,0,0],
         [0,0,1,1+1/z,1,0],
         [0,0,0,1,-1-z,1],
         [1/w,0,0,0,1,1+1/z]])
	# same but rows flipped
    sympyMat2 = sp.Matrix(
        [[1/w,0,0,0,1,1+1/z],
         [1,1+1/z,1,0,0,0],
         [0,1,-1-z,1,0,0],
         [-1-z,1,0,0,0,w],
         [0,0,1,1+1/z,1,0],
         [0,0,0,1,-1-z,1]])
    # same but w <-> z
    sympyMat3 = sp.Matrix(
        [[1/z,0,0,0,1,1+1/w],
         [1,1+1/w,1,0,0,0],
         [0,1,-1-w,1,0,0],
         [-1-w,1,0,0,0,w],
         [0,0,1,1+1/w,1,0],
         [0,0,0,1,-1-w,1]])
    # same but z <-> 1/z
    sympyMat4 = sp.Matrix(
        [[z,0,0,0,1,1+1/w],
         [1,1+1/w,1,0,0,0],
         [0,1,-1-w,1,0,0],
         [-1-w,1,0,0,0,w],
         [0,0,1,1+1/w,1,0],
         [0,0,0,1,-1-w,1]])
    # other toric phase
    sympyMat5 = sp.Matrix(
        [[-z,0,0,0,0,w,1,0,0,0],
        [0,1/z,1,0,0,0,0,1,0,0],
        [0,1,-1-z,1,0,0,0,0,0,0],
        [0,0,1,1,0,0,0,0,1,0],
        [0,0,0,0,-1,1,0,0,0,1],
        [1/w,0,0,0,1,1+1/z,0,0,0,0],
        [0,-1,0,0,0,0,1,1,0,0],
        [-1,0,0,0,0,0,-1,1,0,0],
        [0,0,0,0,-1,0,0,0,1,-(1/z)],
        [0,0,0,-1,0,0,0,0,z,1]])
	# yet another phase
    sympyMat6 = sp.Matrix(
        [[-1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
         [1,1/z,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
         [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0],
         [0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0],
         [0,0,0,0,-z,1,0,0,0,0,0,0,0,1,0,0,0,0],
         [0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0],
         [0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
         [0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
         [0,0,0,0,0,-1,0,0,0,0,1,0,0,0,0,0,0,0],
         [-1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,-1,1/z,1/w,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,-1,0,w,-z,0,0,0,0,0,0],
         [0,0,0,0,-1,0,0,0,0,0,0,0,1,-1,0,0,0,0],
         [0,0,0,-1,0,0,0,0,0,0,0,0,1,1,0,0,0,0],
         [0,0,-1,0,0,0,0,0,0,0,0,0,0,0,1,-1,0,0],
         [0,-1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0],
         [0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,1/z,1],
         [0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,1,-z]])
	# -----------------------------------------------

    # ----- convert matrices to tensorflow tensor -----
    example_matrices = [sympyMat1, sympyMat2, sympyMat3, sympyMat4, sympyMat5, sympyMat6]
    input_matrices = [tf.sparse.to_dense(tf.sparse.expand_dims(tf.sparse.reset_shape(tf.sparse.from_dense(np.array(kasteleynMat2Vector(mat))), new_shape=(MATRIX_SIZE, MATRIX_SIZE, 9)), axis=0)) for mat in example_matrices]

    input_data = tf.concat(input_matrices, axis=0)
    print(input_data.shape)
    # --------------------------------------------------
    
	# Define the ResNet model
    model = ResNet(input_shape=(MATRIX_SIZE,MATRIX_SIZE,9), num_classes=NUM_CLASSES, activation=HP_ACTIVATION)
    
    logdir = 'y60' + '_b' + str(HP_BATCH_SIZE) + '_ac' + HP_ACTIVATION

    # Predict output
    latest_epoch = checkpoint_loader(path='models/ResNet/checkpoints/'+logdir, model=model)
    predictions = model.predict(input_data)
    print(predictions)