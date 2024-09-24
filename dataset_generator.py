# -*- coding: utf-8 -*-
"""
@author: Benterre

This code reads all matrix_mn.json files found in "DimerML/ZmZn_matrices" and does the following
    mn_sparseSD_MT:
        - converts json matrix to sympy matrix and then to Tensorflow sparse array
        - performs seiberg duality on the sparse tensors (using sparseSD()) ITERAMAX number of times
    __main__:
        - saves dense tensors as a Tensorflow Dataset inside "datasets/ms{MAX_SIZE}_it{ITERAMAX}"

The parameters are:
    - ITERAMAX: Maximum number of Seiberg Duals to apply
    - MAX_SIZE: Padding of the output matrix (with shape (MAX_SIZE, MAX_SIZE, 9))
                
"""

import numpy as np
import sympy as sp
import json
import tensorflow as tf
import os
from tqdm import tqdm


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
        # returns zero if polynomial is trivial
        return np.array([0,0,0,0,0,0,0,0,0])
    elif "w" not in str(exp) and "z" not in str(exp):
        # returns integer if the polynomial is order 0
        return int(exp)*np.array([1,0,0,0,0,0,0,0,0])
    else:
        # otherwise cycle through monomials
        monomials = exp.as_ordered_terms()
        vector = np.array([0,0,0,0,0,0,0,0,0])
        for term in monomials:
            term_str = str(term)
            cst = 1
            if "w" not in term_str and "z" not in term_str:
                # return integer if the monomial is order 0
                cst = int(term_str)
                term_vector = np.array([1,0,0,0,0,0,0,0,0])
            else:
                if "w" not in term_str:
                    # find position of z
                    temp_index = term_str.index("z")
                elif "z" not in term_str:
                    # find position of w
                    temp_index = term_str.index("w")
                else:
                    # first first position of w and z
                    temp_index = min(term_str.index("w"),term_str.index("z"))
                
                if temp_index == 0:
                    # if z or w is the first char, then multiplicative constant is 1
                    cst = 1
                elif temp_index == 1 and term_str[0] == "-":
                    # if z or w is the second char, and first char is '-'
                    # then multiplicative constant is -1
                    cst = -1
                elif term_str[temp_index-1] == "/":
                    # if z or w is preceded by '/' then multiplicative constant
                    # is everything before that and we convert term_str to the
                    # standard form '1/z' or '1/w'
                    cst = int(term_str[:temp_index-1])
                    term_str = "1" + term_str[temp_index-1:]
                    temp_index = 0
                elif term_str[temp_index-2] == "/":
                    # if z or w is preceded by '/' two chars down,
                    # then the multiplicative constant is everything before that
                    # and we convert to the standard form '1/(z*w)' or '1/(w*z)'
                    cst = int(term_str[:temp_index-2])
                    term_str = "1" + term_str[temp_index-2:]
                    temp_index = 0
                else:
                    # Otherwise the multiplicative constant is everything before z or w
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

#Find all squares of +1/-1 inside the Tensorflow sparse tensor
def sparseFindSquare(tfsparse):
    # search is performed starting from the first +-1 element (when indices[i][2] == 0)
    # running down the list until we find a +-1 at the same [0] position
    # then again until we find a +-1 at the same [1] position
    # and again until we form a square
    squares = []
    indices = tfsparse.indices.numpy()
    length = len(indices)
    for i in range(length):
        if indices[i][2] == 0:
            for j in range(i+1, length):
                if indices[j][2] == 0 and indices[j][0] == indices[i][0]:
                    for k in range(j+1, length):
                        if indices[k][2] == 0 and indices[k][1] == indices[i][1]:
                            for l in range(k+1, length):
                                if indices[l][2] == 0 and indices[l][1] == indices[j][1] and indices[l][0] == indices[k][0]:
                                    squares.append([i,j,k,l])

    return squares

#Seiberg dualize on a sparse tensor
def sparseSD(tfsparse, square):
    #Performs Seiberg Duality for a square of {1,-1} only! Doesn't work for w,z-dependent terms!!
    indices = tfsparse.indices.numpy()
    values = tfsparse.values.numpy()
    shape = tfsparse.shape

    # Collect 2x2 sub matrix elements and positions
    a = values[square[0]]
    a_pos = indices[square[0]]
    b = values[square[1]]
    c = values[square[2]]
    d = values[square[3]]
    d_pos = indices[square[3]]

    # Delete 2x2 sub matrix from original
    indices = np.delete(indices, square, axis=0)
    values = np.delete(values, square, axis=0)

    # Apply Seiberg duality formula
    indices_to_add = [[a_pos[0],shape[1],0],[d_pos[0],shape[1]+1,0],[shape[0]+1,a_pos[1],0],[shape[0],d_pos[1],0],[shape[0],shape[1],0],[shape[0]+1,shape[1],0],[shape[0],shape[1]+1,0],[shape[0]+1,shape[1]+1,0]]
    values_to_add = [1,1,-1,-1,1/b,1/a,1/d,1/c]

    # Add new index and values to original
    indices = np.concatenate((indices, indices_to_add))
    values = np.concatenate((values, values_to_add))

    # Sort indices and values in increasing order of indices
    sort = np.lexsort((indices[:, 1], indices[:, 0]))
    indices = indices[sort]
    values = values[sort]
    
    # Return sparse tensor with new indices and values
    return tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=(shape[0]+2,shape[1]+2,9))

#Read .json file and performs Seiberg duality
def mn_sparseSD_MT(depthMax=999999, iteraMax=99999, max_size=128, path='ZmZn_matrices/matrix_2_2.json'):
    # Define w and z symbols for Sympy
    w = sp.Symbol('w')
    z = sp.Symbol('z')

    # Read json file
    with open(path, 'r') as file:
        matrices = json.load(file)

    progress_bar = tqdm(total=iteraMax+1, desc="Generating Seiberg Duals")

    # Convert json matrix to tensorflow sparse tensor
    sparse_matrix = tf.sparse.from_dense(kasteleynMat2Vector(sp.Matrix(sp.sympify(matrices[0]))))
    label = matrices[1] # label format: (m,n)

    sparse_sd_matrix_list = [sparse_matrix]
    label_list = [[label[0], label[1], 0]] # third label is depth in Seiberg duality

    progress_bar.update(1)

    squares = sparseFindSquare(sparse_matrix) # find all possible squares to dualise on

    if sparse_sd_matrix_list[0].shape[0] > max_size:
        # if bigger than max size, just skip
        progress_bar.update(iteraMax)
        return [[],[]]
    elif len(squares) == 0:
        # if there are no squares, just return the original matrix (in sparse format)
        progress_bar.update(iteraMax)
        sparse_reshaped = tf.sparse.reset_shape(sparse_matrix, new_shape=(max_size, max_size, 9))
        return [[sparse_reshaped], label_list]
    else:
        # otherwise, iterate through all the possible duals iteramax number of times
        itera = 0
        current_mat = 0
        while itera < iteraMax:

            temp_matrix = sparse_sd_matrix_list[current_mat] # set current working matrix
            temp_labels = label_list[current_mat] # set current working labels

            squares = sparseFindSquare(temp_matrix)
            for square in squares:
                # perform Seiberg duality on each square and append to sparse_sd_matrix_list
                progress_bar.update(1)
                sparse_sd_matrix_list.append(sparseSD(temp_matrix, square))
                label_list.append([temp_labels[0],temp_labels[1],temp_labels[2]+1])

                itera += 1
                if itera > iteraMax:
                    break
                    
            current_mat += 1
            if current_mat > len(sparse_sd_matrix_list):
                # if there are no more matrices to dualise on, end
                progress_bar.update(iteraMax-itera)
                break
            elif label_list[current_mat][2] > depthMax-1:
                progress_bar.update(iteraMax-itera)
                break
            if sparse_sd_matrix_list[current_mat].shape[0] > max_size:
                # break if current matrix is larger than the max size (saves time)
                # Seiberg duality will only make the matrix bigger
                progress_bar.update(iteraMax-itera)
                break
        
        # now cycle through all listed matrices and pad them to the maximal size
        reshaped_list = []
        reshaped_data = []
        for i in range(len(sparse_sd_matrix_list)):
            if sparse_sd_matrix_list[i].shape[0] < max_size:
                reshaped_list.append(tf.sparse.reset_shape(sparse_sd_matrix_list[i], new_shape=(max_size, max_size, 9)))
                reshaped_data.append(label_list[i])
    progress_bar.close()

    return [reshaped_list, reshaped_data]


if __name__ == "__main__":
    # Global Parameters - change to desired values
    ITERAMAX = 1000
    MAX_SIZE = 128

    # Loop over all .json files in partitioned_jsons folder
    for folder in sorted(os.listdir("ZmZn_matrices")):
        if folder != ".DS_Store": # safety check for potential DS_Store files
            M = folder.split("_")[1]
            N = folder.split("_")[2][0:-5]

            if (int(M)*int(N)) >= MAX_SIZE:
                print("M= " + str(M) + ", N=" + str(N) + " matrix too big. Skipping...")
            elif os.path.exists("datasets/ms" + str(MAX_SIZE) + "_it" + str(ITERAMAX) + "/" + str(M) + "_" + str(N) + "_train") and os.path.exists("datasets/ms" + str(MAX_SIZE) + "_it" + str(ITERAMAX) + "/" + str(M) + "_" + str(N) + "_validation"):
                # check if dataset was already generated
                print("M= " + str(M) + ", N=" + str(N) + " datasets already exist. Skipping...")
            else:
                print("Now doing (m,n)=(", M, ",", N,")")
                path = 'ZmZn_matrices/matrix_' + str(M) + '_' + str(N) + '.json'

                # Get list of Seiberg duals following input parameters
                sparse_list, labels = mn_sparseSD_MT(iteraMax=ITERAMAX, max_size=MAX_SIZE, path=path)

                # Recombine output into a single list of tensors (both for sparse and dense)
                sparse_matrices = []
                for i in range(len(labels)):
                    sparse_matrices.append(tf.sparse.expand_dims(sparse_list[i], axis=0))
                    
                print("A total of " + str(len(sparse_matrices)) + " matrices were generated")

                # Convert list of tensors to tf.sparse.SparseTensor
                concatenated_matrices = tf.sparse.concat(sp_inputs=sparse_matrices, axis=0)
                concatenated_labels = tf.sparse.from_dense(tf.convert_to_tensor(labels))

                # Convert tensors to Datasets
                dataset_labels = tf.data.Dataset.from_tensor_slices(concatenated_labels)
                dataset_matrices = tf.data.Dataset.from_tensor_slices(concatenated_matrices)
                dataset = tf.data.Dataset.zip(dataset_matrices, dataset_labels)

                # Determine the size of the dataset
                dataset_size = dataset.cardinality().numpy()
                train_size = int(0.8 * dataset_size)

                # Shuffle the dataset
                dataset = dataset.shuffle(buffer_size=int(dataset_size))

                # Separate training and validation
                train_dataset = dataset.take(train_size)
                validation_dataset = dataset.skip(train_size)

                # Save Datasets
                if not os.path.exists("datasets/ms" + str(MAX_SIZE) + "_it" + str(ITERAMAX) + "/" + str(M) + "_" + str(N) + "_train"):
                    tf.data.Dataset.save(train_dataset, "datasets/ms" + str(MAX_SIZE) + "_it" + str(ITERAMAX) + "/" + str(M) + "_" + str(N) + "_train", compression="GZIP")
                    tf.data.Dataset.save(validation_dataset, "datasets/ms" + str(MAX_SIZE) + "_it" + str(ITERAMAX) + "/" + str(M) + "_" + str(N) + "_validation", compression="GZIP")

                print("m = " + str(M) + ", n = " + str(N) + " saved successfully")