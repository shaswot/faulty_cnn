import tensorflow as tf
import numpy as np
import warnings

# CONSTANTS
from libs.errmatmul import matmul_ERRexpbitflips, N_THREADS_PER_BLOCK, NO_OF_CLASSES

# Batchwise accuracy evaluation of LeNet CNN model using matmul_ERRexpbitflips
@tf.function
def batch_lenet_3hidden_ERRexpbitflips( b_images,
                                        b_labels,
                                        model,
                                        error_profile_c0,
                                        error_profile_h0,
                                        error_profile_h1,
                                        error_profile_h2,
                                        error_profile_op,
                                        ERR_PARAM_TF,
                                        clayer0_shuffle_order,
                                        hlayer0_shuffle_order,
                                        hlayer1_shuffle_order,
                                        hlayer2_shuffle_order,
                                        oplayer_shuffle_order):
    """
        Infer a batch of images/labels using given model, error_profile and shuffle order
        Applies only to 
        > LeNet CNN with THREE hidden layers
        > error injection and shuffling is executed in ALL Layers
        > bitflip in exponent field of FP32
        
        Calls: matmul_ERRexpbitflips
    """
       
    # get weights and biases from model
    conv2d_kernels, conv2d_biases = model.get_layer("conv2d").weights
    fc_0_weights, fc_0_biases = model.get_layer("fc_0").weights
    fc_1_weights, fc_1_biases = model.get_layer("fc_1").weights
    fc_2_weights, fc_2_biases = model.get_layer("fc_2").weights
    op_layer_weights, op_layer_biases = model.get_layer("op_layer").weights
    
    #####################################################################################
    # L0: CONVOLUTION LAYER
    ## L0.A: Get dimension values
    #### kernel height, kernel width, no of channels in input image, no of filter kernels 
    kr_ht, kr_wt, no_ch, no_kr = conv2d_kernels.shape

    no_im = b_images.shape[0]
    no_ch = b_images.shape[-1]
    
    assert no_im == len(b_labels)
    
    ### input image dimensions
    im_ht = b_images.shape[1]
    im_wt = b_images.shape[2]

    ### convolution layer output dimensions (no padding, valid convolution)
    y_ht = im_ht - kr_ht + 1
    y_wt = im_wt - kr_wt + 1
    
    ### patch dimensions
    no_of_patches = y_ht * y_wt
    patch_len     = kr_ht * kr_wt * no_ch
    
    ## L0.B: Extract Images Patches
    patches = tf.image.extract_patches(images=b_images,
                                     sizes=[1, kr_ht, kr_wt, 1],
                                     strides=[1, 1, 1, 1],
                                     rates=[1, 1, 1, 1],
                                     padding='VALID')
    ### flatten patches
    flat_patches = tf.reshape(patches, (no_im, no_of_patches, patch_len))
    ### tranpose for matrix multiplication
    flat_patches = tf.transpose(flat_patches, (0,2,1))

    ## L0.C: Flatten filter kernels
    ### first reorder kernels by no. of output-kernels
    flat_kernels = tf.transpose(conv2d_kernels, perm=(3,0,1,2))
    ### then reshape to required matrix shape
    flat_kernels = tf.reshape(flat_kernels, (no_kr, kr_ht*kr_wt*no_ch))
    
    ## L0.D: Perform Matrix Multiplication
    conv_mul_out_list = []
    ### for each image in batch
    for im in range(no_im):
        single_im_patch = flat_patches[im,:,:]
        # conv_out_list.append(tf.matmul(flat_kernels, single_im_patch))
        BLOCK_HEIGHT = N_THREADS_PER_BLOCK # no. of threads per block
        BLOCK_WIDTH = kr_ht*kr_wt # totcols is always (going to be) a multiple of BLOCK_WIDTH
        BATCH_BLOCK_SIZE = 32 # user-defined: NOT the actual batch block size in this context.
                                # simply the tile block width of matB
        # pad matrix for good matrix shape
        no_cols_to_pad = BATCH_BLOCK_SIZE-(single_im_patch.shape[1]%BATCH_BLOCK_SIZE)
        paddings = tf.constant([[0, 0,], # padding above and below
                                [0, no_cols_to_pad]]) # padding left and right
        padded_single_im_patch = tf.pad(single_im_patch, 
                                        paddings,
                                        mode="CONSTANT", 
                                        constant_values=0.0)
        # is shuffling required
        if clayer0_shuffle_order is not None:
            # shuffle filter order matrix
            shuffled_kernels = tf.gather(flat_kernels, clayer0_shuffle_order)
        else:
            shuffled_kernels = flat_kernels
        
        # is error injection required
        if error_profile_c0 is not None:
            shuffled_conv_mul_out = matmul_ERRexpbitflips(shuffled_kernels, 
                                                           padded_single_im_patch,
                                                           BLOCK_HEIGHT, 
                                                           BLOCK_WIDTH, 
                                                           BATCH_BLOCK_SIZE, 
                                                           ERR_PROFILE=error_profile_c0,
                                                           ERR_PARAM_TF=ERR_PARAM_TF,)[:,:-no_cols_to_pad]            

        else:
            shuffled_conv_mul_out = tf.matmul(shuffled_kernels, padded_single_im_patch)[:,:-no_cols_to_pad]
        
        # was the kernel matrix shuffled ?
        if clayer0_shuffle_order is not None:
            # unshuffle conv_out
            indices = tf.expand_dims(clayer0_shuffle_order, axis=1)
            updates = tf.range(tf.size(indices))
            shape = clayer0_shuffle_order.shape
            scatter = tf.scatter_nd(indices, updates, shape)
            conv_mul_out = tf.gather(shuffled_conv_mul_out, scatter)
        else:
            conv_mul_out = shuffled_conv_mul_out
        conv_mul_out_list.append(conv_mul_out)
        # this completes the matrix multiplication equivalent of convolution of *ONE* image in the batch of image
        
    conv_out = tf.stack(conv_mul_out_list)
    conv_out = tf.transpose(conv_out, (0,2,1)) # rearrange channel order
    conv_out = tf.reshape(conv_out, (no_im, y_ht,y_wt, no_kr)) # reshape to filter output shape

    ## Add bias
    conv_out = tf.nn.bias_add(conv_out, conv2d_biases)
    ## ReLU
    conv_out = tf.nn.relu(conv_out)
    #####################################################################################
    
    # L1: MAX POOLING LAYER
    pool_out = tf.nn.max_pool(conv_out,
                                ksize=[1, 2, 2, 1], #(batch_size, height, width, depth)
                                strides=[1, 2, 2, 1], #(batch_size, height, width, depth)
                                padding='VALID')
    #####################################################################################
    
    # L2: FLATTEN LAYER
    flat_out = tf.reshape(pool_out, (no_im, -1) ) #[batch_size, flat_vec_size]
    
    #####################################################################################
    
    # L3: FC0
    ## tranpose input vector
    fc_0_in = tf.transpose(flat_out, perm=[1,0]) #[flat_vec_size, batch_size]
    ## transpose weight matrices
    fc_0_weights_tr = tf.transpose(fc_0_weights, perm=[1,0]) #[no_of_weights, flat_vec_size]

    ## is shuffling required
    if hlayer0_shuffle_order is not None:
        ## shuffle weight matrix
        shuffled_weights = tf.gather(fc_0_weights_tr, hlayer0_shuffle_order)
    else:
        shuffled_weights = fc_0_weights_tr
        
    ## is error injection required
    if error_profile_h0 is not None:
        ## multiply with shuffled weight matrix
        BLOCK_HEIGHT = N_THREADS_PER_BLOCK # no. of threads per block
        BLOCK_WIDTH = 32 # totcols is always (going to be) a multiple of BLOCK_WIDTH
        BATCH_BLOCK_SIZE = 1 # in reality, inference is always one image at a time. 
                             # However, here we are using batch inference here for speedup
        shuffled_mult_out = matmul_ERRexpbitflips(shuffled_weights, 
                                                   fc_0_in,
                                                   BLOCK_HEIGHT, 
                                                   BLOCK_WIDTH, 
                                                   BATCH_BLOCK_SIZE, 
                                                   ERR_PROFILE=error_profile_h0,
                                                   ERR_PARAM_TF=ERR_PARAM_TF)
    else:
        shuffled_mult_out = tf.linalg.matmul(shuffled_weights, fc_0_in)
        
    ## was the weight matrix shuffled
    if hlayer0_shuffle_order is not None:
        # unshuffle mult_out
        indices = tf.expand_dims(hlayer0_shuffle_order, axis=1)
        updates = tf.range(tf.size(indices))
        shape = hlayer0_shuffle_order.shape
        scatter = tf.scatter_nd(indices, updates, shape)
        fc_0_mult_out = tf.gather(shuffled_mult_out, scatter)
    else:
        fc_0_mult_out = shuffled_mult_out
        

    # Add bias
    fc_0_bout = tf.add(fc_0_mult_out, tf.expand_dims(fc_0_biases,axis=1))
    # RelU
    fc_0_out = tf.nn.relu(fc_0_bout)
    # fc_0_out needs to be transposed again in fc_1_in
    # so although fc_0_out shape is not "standard", we output it as it is
    
    #####################################################################################
    
    # L4: FC1
    ## tranpose input vector
    fc_1_in = fc_0_out
    ## transpose weight matrices
    fc_1_weights_tr = tf.transpose(fc_1_weights, perm=[1,0]) #[no_of_weights, flat_vec_size]

    ## is shuffling required
    if hlayer1_shuffle_order is not None:
        ## shuffle weight matrix
        shuffled_weights = tf.gather(fc_1_weights_tr, hlayer1_shuffle_order)
    else:
        shuffled_weights = fc_1_weights_tr
        
    ## is error injection required
    if error_profile_h1 is not None:
        ## multiply with shuffled weight matrix
        BLOCK_HEIGHT = N_THREADS_PER_BLOCK # no. of threads per block
        BLOCK_WIDTH = 32 # totcols is always (going to be) a multiple of BLOCK_WIDTH
        BATCH_BLOCK_SIZE = 1 # inference is always one image at a time.
        shuffled_mult_out = matmul_ERRexpbitflips(shuffled_weights, 
                                                   fc_1_in,
                                                   BLOCK_HEIGHT, 
                                                   BLOCK_WIDTH, 
                                                   BATCH_BLOCK_SIZE, 
                                                   ERR_PROFILE=error_profile_h1,
                                                   ERR_PARAM_TF=ERR_PARAM_TF)
    else:
        shuffled_mult_out = tf.linalg.matmul(shuffled_weights, fc_1_in)
        
    ## was the weight matrix shuffled
    if hlayer1_shuffle_order is not None:
        # unshuffle mult_out
        indices = tf.expand_dims(hlayer1_shuffle_order, axis=1)
        updates = tf.range(tf.size(indices))
        shape = hlayer1_shuffle_order.shape
        scatter = tf.scatter_nd(indices, updates, shape)
        fc_1_mult_out = tf.gather(shuffled_mult_out, scatter)
    else:
        fc_1_mult_out = shuffled_mult_out
        

    # Add bias
    fc_1_bout = tf.add(fc_1_mult_out, tf.expand_dims(fc_1_biases,axis=1))
    # RelU
    fc_1_out = tf.nn.relu(fc_1_bout)
    # fc_1_out needs to be transposed again in fc_2_in
    # so although fc_1_out shape is not "standard", we output it as it is
    
#####################################################################################
    
    # L5: FC2
    ## tranpose input vector
    fc_2_in = fc_1_out
    ## transpose weight matrices
    fc_2_weights_tr = tf.transpose(fc_2_weights, perm=[1,0]) #[no_of_weights, flat_vec_size]

    ## is shuffling required
    if hlayer2_shuffle_order is not None:
        ## shuffle weight matrix
        shuffled_weights = tf.gather(fc_2_weights_tr, hlayer2_shuffle_order)
    else:
        shuffled_weights = fc_2_weights_tr
        
    ## is error injection required
    if error_profile_h2 is not None:
        ## multiply with shuffled weight matrix
        BLOCK_HEIGHT = N_THREADS_PER_BLOCK # no. of threads per block
        BLOCK_WIDTH = 32 # totcols is always (going to be) a multiple of BLOCK_WIDTH
        BATCH_BLOCK_SIZE = 1 # inference is always one image at a time.
        shuffled_mult_out = matmul_ERRexpbitflips(shuffled_weights, 
                                                   fc_2_in,
                                                   BLOCK_HEIGHT, 
                                                   BLOCK_WIDTH, 
                                                   BATCH_BLOCK_SIZE, 
                                                   ERR_PROFILE=error_profile_h2,
                                                   ERR_PARAM_TF=ERR_PARAM_TF)
    else:
        shuffled_mult_out = tf.linalg.matmul(shuffled_weights, fc_2_in)
        
    ## was the weight matrix shuffled
    if hlayer2_shuffle_order is not None:
        # unshuffle mult_out
        indices = tf.expand_dims(hlayer2_shuffle_order, axis=1)
        updates = tf.range(tf.size(indices))
        shape = hlayer2_shuffle_order.shape
        scatter = tf.scatter_nd(indices, updates, shape)
        fc_2_mult_out = tf.gather(shuffled_mult_out, scatter)
    else:
        fc_2_mult_out = shuffled_mult_out
        

    # Add bias
    fc_2_bout = tf.add(fc_2_mult_out, tf.expand_dims(fc_2_biases,axis=1))
    # RelU
    fc_2_out = tf.nn.relu(fc_2_bout)
    # fc_1_out needs to be transposed again in fc_2_in
    # so although fc_1_out shape is not "standard", we output it as it is
    
#####################################################################################
    
    # L6: OUTPUT LAYER
    ## tranpose input vector
    op_layer_in = fc_2_out
    ## transpose weight matrices
    op_layer_weights_tr = tf.transpose(op_layer_weights, perm=[1,0]) #[no_of_weights, flat_vec_size]

    ## is shuffling required
    if oplayer_shuffle_order is not None:
        ## shuffle weight matrix
        shuffled_weights = tf.gather(op_layer_weights_tr, oplayer_shuffle_order)
    else:
        shuffled_weights = op_layer_weights_tr
        
    ## is error injection required
    if error_profile_op is not None:
        ## multiply with shuffled weight matrix
        BLOCK_HEIGHT = NO_OF_CLASSES # no. of threads per block
        BLOCK_WIDTH = 32 # totcols is always (going to be) a multiple of BLOCK_WIDTH
        BATCH_BLOCK_SIZE = 1 # inference is always one image at a time.
        shuffled_mult_out = matmul_ERRexpbitflips(shuffled_weights, 
                                                   op_layer_in,
                                                   BLOCK_HEIGHT, 
                                                   BLOCK_WIDTH, 
                                                   BATCH_BLOCK_SIZE, 
                                                   ERR_PROFILE=error_profile_op,
                                                   ERR_PARAM_TF=ERR_PARAM_TF)
    else:
        shuffled_mult_out = tf.linalg.matmul(shuffled_weights, op_layer_in)
        
    ## was the weight matrix shuffled
    if oplayer_shuffle_order is not None:
        # unshuffle mult_out
        indices = tf.expand_dims(oplayer_shuffle_order, axis=1)
        updates = tf.range(tf.size(indices))
        shape = oplayer_shuffle_order.shape
        scatter = tf.scatter_nd(indices, updates, shape)
        op_layer_mult_out = tf.gather(shuffled_mult_out, scatter)
    else:
        op_layer_mult_out = shuffled_mult_out
        

    # Add bias
    op_layer_bout = tf.add(op_layer_mult_out, tf.expand_dims(op_layer_biases,axis=1))
    # Softmax
    op_layer_out = tf.nn.softmax(op_layer_bout, axis=0)
    # Tranpose to standard order
    class_scores = tf.transpose(op_layer_out, perm=[1,0])
    
    # Get predictions
    predictions = tf.math.argmax(class_scores, axis=1)
    ## count no. of wrong predicitons
    # return predictions
    return tf.math.reduce_sum(tf.cast(tf.math.not_equal(tf.cast(b_labels, dtype=tf.int64), predictions), dtype=tf.int64))
###############################################################################################################################