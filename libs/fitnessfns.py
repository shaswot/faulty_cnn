import tensorflow as tf
import numpy as np
import warnings

# CONSTANTS
from libs.errmatmul import matmul_ERR, N_THREADS_PER_BLOCK
NO_OF_CLASSES = 10

# Batchwise accuracy evaluation of mnist32_cnn model using matmul_ERR
@tf.function
def batch_mnist32_cnn_ERR(b_images,
                        b_labels,
                        model,
                        error_profile_c0,
                        error_profile_h0,
                        error_profile_h1,
                        error_profile_h2,
                        error_profile_op,
                        ERR_PARAM_TF,
                        shuffle_order_c0,
                        shuffle_order_h0,
                        shuffle_order_h1,
                        shuffle_order_h2,
                        shuffle_order_op):
    """
        Infer a batch of images/labels using given model, error_profile and shuffle order
        Applies only to 
        > mnist32_cnn with THREE hidden layers
        > error injection and shuffling is executed in ALL Layers
        > bitflip in exponent field of FP32 [ERR: 0,1,-1]
        > truncation in mantissa field of FP32 [ERR: 2, 3]
        
        Calls: matmul_ERR
    """
       
    # get weights and biases from model
    c0_kernels, c0_biases = model.get_layer("c0").weights
    h0_weights, h0_biases = model.get_layer("h0").weights
    h1_weights, h1_biases = model.get_layer("h1").weights
    h2_weights, h2_biases = model.get_layer("h2").weights
    op_weights, op_biases = model.get_layer("op").weights
    
    #####################################################################################
    # L0: CONVOLUTION LAYER
    ## L0.A: Get dimension values
    #### kernel height, kernel width, no of channels in input image, no of filter kernels 
    kr_ht, kr_wt, no_ch, no_kr = c0_kernels.shape

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
    flat_kernels = tf.transpose(c0_kernels, perm=(3,0,1,2))
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
        if shuffle_order_c0 is not None:
            # shuffle filter order matrix
            shuffled_kernels = tf.gather(flat_kernels, shuffle_order_c0)
        else:
            shuffled_kernels = flat_kernels
        
        # is error injection required
        if error_profile_c0 is not None:
            shuffled_conv_mul_out = matmul_ERR(shuffled_kernels, 
                                               padded_single_im_patch,
                                               BLOCK_HEIGHT, 
                                               BLOCK_WIDTH, 
                                               BATCH_BLOCK_SIZE, 
                                               ERR_PROFILE=error_profile_c0,
                                               ERR_PARAM_TF=ERR_PARAM_TF,)[:,:-no_cols_to_pad]            

        else:
            shuffled_conv_mul_out = tf.matmul(shuffled_kernels, padded_single_im_patch)[:,:-no_cols_to_pad]
        
        # was the kernel matrix shuffled ?
        if shuffle_order_c0 is not None:
            # unshuffle conv_out
            indices = tf.expand_dims(shuffle_order_c0, axis=1)
            updates = tf.range(tf.size(indices))
            shape = shuffle_order_c0.shape
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
    conv_out = tf.nn.bias_add(conv_out, c0_biases)
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
    h0_in = tf.transpose(flat_out, perm=[1,0]) #[flat_vec_size, batch_size]
    ## transpose weight matrices
    h0_weights_tr = tf.transpose(h0_weights, perm=[1,0]) #[no_of_weights, flat_vec_size]

    ## is shuffling required
    if shuffle_order_h0 is not None:
        ## shuffle weight matrix
        shuffled_weights = tf.gather(h0_weights_tr, shuffle_order_h0)
    else:
        shuffled_weights = h0_weights_tr
        
    ## is error injection required
    if error_profile_h0 is not None:
        ## multiply with shuffled weight matrix
        BLOCK_HEIGHT = N_THREADS_PER_BLOCK # no. of threads per block
        BLOCK_WIDTH = 32 # totcols is always (going to be) a multiple of BLOCK_WIDTH
        BATCH_BLOCK_SIZE = 32 # in reality, inference is always one image at a time. 
                             # However, here we are using batch inference here for speedup
        shuffled_mult_out = matmul_ERR(shuffled_weights, 
                                       h0_in,
                                       BLOCK_HEIGHT, 
                                       BLOCK_WIDTH, 
                                       BATCH_BLOCK_SIZE, 
                                       ERR_PROFILE=error_profile_h0,
                                       ERR_PARAM_TF=ERR_PARAM_TF)
    else:
        shuffled_mult_out = tf.linalg.matmul(shuffled_weights, h0_in)
        
    ## was the weight matrix shuffled
    if shuffle_order_h0 is not None:
        # unshuffle mult_out
        indices = tf.expand_dims(shuffle_order_h0, axis=1)
        updates = tf.range(tf.size(indices))
        shape = shuffle_order_h0.shape
        scatter = tf.scatter_nd(indices, updates, shape)
        h0_mult_out = tf.gather(shuffled_mult_out, scatter)
    else:
        h0_mult_out = shuffled_mult_out
        

    # Add bias
    h0_bout = tf.add(h0_mult_out, tf.expand_dims(h0_biases,axis=1))
    # RelU
    h0_out = tf.nn.relu(h0_bout)
    # h0_out needs to be transposed again in h1_in
    # so although h0_out shape is not "standard", we output it as it is
    
    #####################################################################################
    
    # L4: FC1
    ## tranpose input vector
    h1_in = h0_out
    ## transpose weight matrices
    h1_weights_tr = tf.transpose(h1_weights, perm=[1,0]) #[no_of_weights, flat_vec_size]

    ## is shuffling required
    if shuffle_order_h1 is not None:
        ## shuffle weight matrix
        shuffled_weights = tf.gather(h1_weights_tr, shuffle_order_h1)
    else:
        shuffled_weights = h1_weights_tr
        
    ## is error injection required
    if error_profile_h1 is not None:
        ## multiply with shuffled weight matrix
        BLOCK_HEIGHT = N_THREADS_PER_BLOCK # no. of threads per block
        BLOCK_WIDTH = 32 # totcols is always (going to be) a multiple of BLOCK_WIDTH
        BATCH_BLOCK_SIZE = 32 # inference is always one image at a time.
        shuffled_mult_out = matmul_ERR(shuffled_weights, 
                                       h1_in,
                                       BLOCK_HEIGHT, 
                                       BLOCK_WIDTH, 
                                       BATCH_BLOCK_SIZE, 
                                       ERR_PROFILE=error_profile_h1,
                                       ERR_PARAM_TF=ERR_PARAM_TF)
    else:
        shuffled_mult_out = tf.linalg.matmul(shuffled_weights, h1_in)
        
    ## was the weight matrix shuffled
    if shuffle_order_h1 is not None:
        # unshuffle mult_out
        indices = tf.expand_dims(shuffle_order_h1, axis=1)
        updates = tf.range(tf.size(indices))
        shape = shuffle_order_h1.shape
        scatter = tf.scatter_nd(indices, updates, shape)
        h1_mult_out = tf.gather(shuffled_mult_out, scatter)
    else:
        h1_mult_out = shuffled_mult_out
        

    # Add bias
    h1_bout = tf.add(h1_mult_out, tf.expand_dims(h1_biases,axis=1))
    # RelU
    h1_out = tf.nn.relu(h1_bout)
    # h1_out needs to be transposed again in h2_in
    # so although h1_out shape is not "standard", we output it as it is
    
#####################################################################################
    
    # L5: FC2
    ## tranpose input vector
    h2_in = h1_out
    ## transpose weight matrices
    h2_weights_tr = tf.transpose(h2_weights, perm=[1,0]) #[no_of_weights, flat_vec_size]

    ## is shuffling required
    if shuffle_order_h2 is not None:
        ## shuffle weight matrix
        shuffled_weights = tf.gather(h2_weights_tr, shuffle_order_h2)
    else:
        shuffled_weights = h2_weights_tr
        
    ## is error injection required
    if error_profile_h2 is not None:
        ## multiply with shuffled weight matrix
        BLOCK_HEIGHT = N_THREADS_PER_BLOCK # no. of threads per block
        BLOCK_WIDTH = 32 # totcols is always (going to be) a multiple of BLOCK_WIDTH
        BATCH_BLOCK_SIZE = 32 # inference is always one image at a time.
        shuffled_mult_out = matmul_ERR(shuffled_weights, 
                                       h2_in,
                                       BLOCK_HEIGHT, 
                                       BLOCK_WIDTH, 
                                       BATCH_BLOCK_SIZE, 
                                       ERR_PROFILE=error_profile_h2,
                                       ERR_PARAM_TF=ERR_PARAM_TF)
    else:
        shuffled_mult_out = tf.linalg.matmul(shuffled_weights, h2_in)
        
    ## was the weight matrix shuffled
    if shuffle_order_h2 is not None:
        # unshuffle mult_out
        indices = tf.expand_dims(shuffle_order_h2, axis=1)
        updates = tf.range(tf.size(indices))
        shape = shuffle_order_h2.shape
        scatter = tf.scatter_nd(indices, updates, shape)
        h2_mult_out = tf.gather(shuffled_mult_out, scatter)
    else:
        h2_mult_out = shuffled_mult_out
        

    # Add bias
    h2_bout = tf.add(h2_mult_out, tf.expand_dims(h2_biases,axis=1))
    # RelU
    h2_out = tf.nn.relu(h2_bout)
    # h1_out needs to be transposed again in h2_in
    # so although h1_out shape is not "standard", we output it as it is
    
#####################################################################################
    
    # L6: OUTPUT LAYER
    ## tranpose input vector
    op_in = h2_out
    ## transpose weight matrices
    op_weights_tr = tf.transpose(op_weights, perm=[1,0]) #[no_of_weights, flat_vec_size]

    ## is shuffling required
    if shuffle_order_op is not None:
        ## shuffle weight matrix
        shuffled_weights = tf.gather(op_weights_tr, shuffle_order_op)
    else:
        shuffled_weights = op_weights_tr
        
    ## is error injection required
    if error_profile_op is not None:
        ## multiply with shuffled weight matrix
        BLOCK_HEIGHT = NO_OF_CLASSES # no. of threads per block
        BLOCK_WIDTH = 32 # totcols is always (going to be) a multiple of BLOCK_WIDTH
        BATCH_BLOCK_SIZE = 32 # inference is always one image at a time.
        shuffled_mult_out = matmul_ERR(shuffled_weights, 
                                       op_in,
                                       BLOCK_HEIGHT, 
                                       BLOCK_WIDTH, 
                                       BATCH_BLOCK_SIZE, 
                                       ERR_PROFILE=error_profile_op,
                                       ERR_PARAM_TF=ERR_PARAM_TF)
    else:
        shuffled_mult_out = tf.linalg.matmul(shuffled_weights, op_in)
        
    ## was the weight matrix shuffled
    if shuffle_order_op is not None:
        # unshuffle mult_out
        indices = tf.expand_dims(shuffle_order_op, axis=1)
        updates = tf.range(tf.size(indices))
        shape = shuffle_order_op.shape
        scatter = tf.scatter_nd(indices, updates, shape)
        op_mult_out = tf.gather(shuffled_mult_out, scatter)
    else:
        op_mult_out = shuffled_mult_out
        

    # Add bias
    op_bout = tf.add(op_mult_out, tf.expand_dims(op_biases,axis=1))
    # Softmax
    op_out = tf.nn.softmax(op_bout, axis=0)
    # Tranpose to standard order
    class_scores = tf.transpose(op_out, perm=[1,0])
    
    # Get predictions
    predictions = tf.math.argmax(class_scores, axis=1)
    ## count no. of wrong predicitons
    # return predictions
    return tf.math.reduce_sum(tf.cast(tf.math.not_equal(tf.cast(b_labels, dtype=tf.int64), predictions), dtype=tf.int64))
##################################################
def ff_mnist32_cnn_ERR(model,
                    error_profile_c0,
                    error_profile_h0,
                    error_profile_h1,
                    error_profile_h2,
                    error_profile_op,
                    ERR_PARAM,
                    shuffle_order_c0,
                    shuffle_order_h0,
                    shuffle_order_h1,
                    shuffle_order_h2,
                    shuffle_order_op,
                    test_set,
                    batchsize):

    if ERR_PARAM is not None:
        ERR_PARAM_TF = tf.constant(ERR_PARAM)
    else:
        ERR_PARAM_TF = None
    # create dataset
    test_dataset = tf.data.Dataset.from_tensor_slices(test_set)
    # Dataset size should be a multiple of BATCH_SIZE
    test_dataset = test_dataset.batch(batchsize, drop_remainder=True)
    tot_images = test_dataset.cardinality() *batchsize
    misses = 0
    for step, (b_images, b_labels) in enumerate(test_dataset):
        b_misses = batch_mnist32_cnn_ERR( b_images,
                                        b_labels,
                                        model,
                                        error_profile_c0,
                                        error_profile_h0,
                                        error_profile_h1,
                                        error_profile_h2,
                                        error_profile_op,
                                        ERR_PARAM_TF,
                                        shuffle_order_c0,
                                        shuffle_order_h0,
                                        shuffle_order_h1,
                                        shuffle_order_h2,
                                        shuffle_order_op)

        misses = misses + b_misses

    return (tot_images-misses)/tot_images
##################################################
def eval_mnist32_cnn_ERR( model,
                        error_profile_c0,
                        error_profile_h0,
                        error_profile_h1,
                        error_profile_h2,
                        error_profile_op,
                        ERR_PARAM,
                        shuffle_order_c0,
                        shuffle_order_h0,
                        shuffle_order_h1,
                        shuffle_order_h2,
                        shuffle_order_op,
                        test_set):
    N_RUNS_PER_SHUFF_ORDER = 3
    BATCHSIZE = 128
    shuffle_result = []
    # Evaluate N_RUNS_PER_SHUFF_ORDER times for each shuffle order
    for i in range(N_RUNS_PER_SHUFF_ORDER):
        accuracy = ff_mnist32_cnn_ERR(model,
                                        error_profile_c0,
                                        error_profile_h0,
                                        error_profile_h1,
                                        error_profile_h2,
                                        error_profile_op,
                                        ERR_PARAM,
                                        shuffle_order_c0,
                                        shuffle_order_h0,
                                        shuffle_order_h1,
                                        shuffle_order_h2,
                                        shuffle_order_op,
                                        test_set,
                                        BATCHSIZE).numpy()
        shuffle_result.append(accuracy)
    return (np.mean(shuffle_result), 
            np.std(shuffle_result))    
##################################################
##################################################
# Batchwise accuracy evaluation of fashion_cnn2 model using matmul_ERR
@tf.function
def batch_fashion_cnn2_ERR(b_images,
                            b_labels,
                            model,
                            error_profile_c0,
                            error_profile_c1,
                            error_profile_h0,
                            error_profile_op,
                            ERR_PARAM_TF,
                            shuffle_order_c0,
                            shuffle_order_c1,
                            shuffle_order_h0,
                            shuffle_order_op):
    """
        Infer a batch of images/labels using given model, error_profile and shuffle order
        Applies only to 
        > fashion_cnn2 with 2 convolution layers and 1 hidden layer
        > error injection and shuffling is executed in ALL Layers
        > bitflip in exponent field of FP32 [ERR: 0,1,-1]
        > truncation in mantissa field of FP32 [ERR: 2, 3]
        
        Calls: matmul_ERR
    """
    # get weights and biases from model
    c0_kernels, c0_biases = model.get_layer("c0").weights
    c1_kernels, c1_biases = model.get_layer("c1").weights
    h0_weights, h0_biases = model.get_layer("h0").weights
    op_weights, op_biases = model.get_layer("op").weights
    
    #####################################################################################
    # L0: CONVOLUTION LAYER c0
    ## L0.A: Get dimension values
    #### kernel height, kernel width, no of channels in input image, no of filter kernels 
    kr_ht, kr_wt, no_ch, no_kr = c0_kernels.shape

    no_im = b_images.shape[0]
    no_ch = b_images.shape[-1]

    assert no_im == len(b_labels)

    ### input image dimensions
    im_ht = b_images.shape[1]
    im_wt = b_images.shape[2]

    ### convolution layer output dimensions (padding=same)
    y_ht = im_ht
    y_wt = im_wt

    ### patch dimensions
    no_of_patches = y_ht * y_wt
    patch_len     = kr_ht * kr_wt * no_ch

    ## L0.B: Extract Images Patches
    patches = tf.image.extract_patches(images=b_images,
                                     sizes=[1, kr_ht, kr_wt, 1],
                                     strides=[1, 1, 1, 1],
                                     rates=[1, 1, 1, 1],
                                     padding='SAME')
    ### flatten patches
    flat_patches = tf.reshape(patches, (no_im, no_of_patches, patch_len))
    ### tranpose for matrix multiplication
    flat_patches = tf.transpose(flat_patches, (0,2,1))

    ## L0.C: Flatten filter kernels
    ### first reorder kernels by no. of output-kernels
    flat_kernels = tf.transpose(c0_kernels, perm=(3,0,1,2))
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
        if shuffle_order_c0 is not None:
            # shuffle filter order matrix
            shuffled_kernels = tf.gather(flat_kernels, shuffle_order_c0)
        else:
            shuffled_kernels = flat_kernels

        # is error injection required
        if error_profile_c0 is not None:
            shuffled_conv_mul_out = matmul_ERR(shuffled_kernels, 
                                               padded_single_im_patch,
                                               BLOCK_HEIGHT, 
                                               BLOCK_WIDTH, 
                                               BATCH_BLOCK_SIZE, 
                                               ERR_PROFILE=error_profile_c0,
                                               ERR_PARAM_TF=ERR_PARAM_TF,)[:,:-no_cols_to_pad]            

        else:
            shuffled_conv_mul_out = tf.matmul(shuffled_kernels, padded_single_im_patch)[:,:-no_cols_to_pad]

        # was the kernel matrix shuffled ?
        if shuffle_order_c0 is not None:
            # unshuffle conv_out
            indices = tf.expand_dims(shuffle_order_c0, axis=1)
            updates = tf.range(tf.size(indices))
            shape = shuffle_order_c0.shape
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
    conv_out = tf.nn.bias_add(conv_out, c0_biases)
    ## ReLU
    conv0_out = tf.nn.relu(conv_out)
    #####################################################################################
    
    # L1: MAX POOLING LAYER
    pool0_out = tf.nn.max_pool(conv0_out,
                                ksize=[1, 2, 2, 1], #(batch_size, height, width, depth)
                                strides=[1, 2, 2, 1], #(batch_size, height, width, depth)
                                padding='VALID')
    #####################################################################################
        
    # L2: DROPOUT LAYER (Disabled in Inference)
    # L3: CONVOLUTION LAYER c1
    ## L3.A: Get dimension values
    #### kernel height, kernel width, no of channels in input image, no of filter kernels 
    kr_ht, kr_wt, no_ch, no_kr = c1_kernels.shape

    no_im = pool0_out.shape[0]
    no_ch = pool0_out.shape[-1]

    assert no_im == len(b_labels)

    ### input image dimensions
    im_ht = pool0_out.shape[1]
    im_wt = pool0_out.shape[2]

    ### convolution layer output dimensions (padding=same)
    y_ht = im_ht
    y_wt = im_wt

    ### patch dimensions
    no_of_patches = y_ht * y_wt
    patch_len     = kr_ht * kr_wt * no_ch

    ## L3.B: Extract Images Patches
    patches = tf.image.extract_patches(images=pool0_out,
                                     sizes=[1, kr_ht, kr_wt, 1],
                                     strides=[1, 1, 1, 1],
                                     rates=[1, 1, 1, 1],
                                     padding='SAME')
    ### flatten patches
    flat_patches = tf.reshape(patches, (no_im, no_of_patches, patch_len))
    ### tranpose for matrix multiplication
    flat_patches = tf.transpose(flat_patches, (0,2,1))

    ## L3.C: Flatten filter kernels
    ### first reorder kernels by no. of output-kernels
    flat_kernels = tf.transpose(c1_kernels, perm=(3,0,1,2))
    ### then reshape to required matrix shape
    flat_kernels = tf.reshape(flat_kernels, (no_kr, kr_ht*kr_wt*no_ch))

    ## L3.D: Perform Matrix Multiplication
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
        if shuffle_order_c1 is not None:
            # shuffle filter order matrix
            shuffled_kernels = tf.gather(flat_kernels, shuffle_order_c1)
        else:
            shuffled_kernels = flat_kernels

        # is error injection required
        if error_profile_c1 is not None:
            shuffled_conv_mul_out = matmul_ERR(shuffled_kernels, 
                                               padded_single_im_patch,
                                               BLOCK_HEIGHT, 
                                               BLOCK_WIDTH, 
                                               BATCH_BLOCK_SIZE, 
                                               ERR_PROFILE=error_profile_c1,
                                               ERR_PARAM_TF=ERR_PARAM_TF,)[:,:-no_cols_to_pad]            

        else:
            shuffled_conv_mul_out = tf.matmul(shuffled_kernels, padded_single_im_patch)[:,:-no_cols_to_pad]

        # was the kernel matrix shuffled ?
        if shuffle_order_c1 is not None:
            # unshuffle conv_out
            indices = tf.expand_dims(shuffle_order_c1, axis=1)
            updates = tf.range(tf.size(indices))
            shape = shuffle_order_c1.shape
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
    conv_out = tf.nn.bias_add(conv_out, c1_biases)
    ## ReLU
    conv1_out = tf.nn.relu(conv_out)
    #####################################################################################
    
    # L4: MAX POOLING LAYER
    pool1_out = tf.nn.max_pool(conv1_out,
                                ksize=[1, 2, 2, 1], #(batch_size, height, width, depth)
                                strides=[1, 2, 2, 1], #(batch_size, height, width, depth)
                                padding='VALID')
    #####################################################################################
    
    # L5: DROPOUT LAYER (Disabled in Inference)
    # L6: FLATTEN LAYER
    flat_out = tf.reshape(pool1_out, (no_im, -1) ) #[batch_size, flat_vec_size]
    #####################################################################################
    
    # L7: HIDDEN LAYER 0
    ## tranpose input vector
    h0_in = tf.transpose(flat_out, perm=[1,0]) #[flat_vec_size, batch_size]
    ## transpose weight matrices
    h0_weights_tr = tf.transpose(h0_weights, perm=[1,0]) #[no_of_weights, flat_vec_size]

    ## is shuffling required
    if shuffle_order_h0 is not None:
        ## shuffle weight matrix
        shuffled_weights = tf.gather(h0_weights_tr, shuffle_order_h0)
    else:
        shuffled_weights = h0_weights_tr

    ## is error injection required
    if error_profile_h0 is not None:
        ## multiply with shuffled weight matrix
        BLOCK_HEIGHT = N_THREADS_PER_BLOCK # no. of threads per block
        BLOCK_WIDTH = 32 # totcols is always (going to be) a multiple of BLOCK_WIDTH
        BATCH_BLOCK_SIZE = 32 # in reality, inference is always one image at a time. 
                             # However, here we are using batch inference here for speedup
        shuffled_mult_out = matmul_ERR(shuffled_weights, 
                                       h0_in,
                                       BLOCK_HEIGHT, 
                                       BLOCK_WIDTH, 
                                       BATCH_BLOCK_SIZE, 
                                       ERR_PROFILE=error_profile_h0,
                                       ERR_PARAM_TF=ERR_PARAM_TF)
    else:
        shuffled_mult_out = tf.linalg.matmul(shuffled_weights, h0_in)

    ## was the weight matrix shuffled
    if shuffle_order_h0 is not None:
        # unshuffle mult_out
        indices = tf.expand_dims(shuffle_order_h0, axis=1)
        updates = tf.range(tf.size(indices))
        shape = shuffle_order_h0.shape
        scatter = tf.scatter_nd(indices, updates, shape)
        h0_mult_out = tf.gather(shuffled_mult_out, scatter)
    else:
        h0_mult_out = shuffled_mult_out


    # Add bias
    h0_bout = tf.add(h0_mult_out, tf.expand_dims(h0_biases,axis=1))
    # RelU
    h0_out = tf.nn.relu(h0_bout)
    # h0_out needs to be transposed again in h1_in
    # so although h0_out shape is not "standard", we output it as it is

    #####################################################################################
    
    # L8: DROPOUT LAYER (Disabled in Inference)
    # L9: OUTPUT LAYER
    ## tranpose input vector
    op_in = h0_out
    ## transpose weight matrices
    op_weights_tr = tf.transpose(op_weights, perm=[1,0]) #[no_of_weights, flat_vec_size]

    ## is shuffling required
    if shuffle_order_op is not None:
        ## shuffle weight matrix
        shuffled_weights = tf.gather(op_weights_tr, shuffle_order_op)
    else:
        shuffled_weights = op_weights_tr

    ## is error injection required
    if error_profile_op is not None:
        ## multiply with shuffled weight matrix
        BLOCK_HEIGHT = NO_OF_CLASSES # no. of threads per block
        BLOCK_WIDTH = 32 # totcols is always (going to be) a multiple of BLOCK_WIDTH
        BATCH_BLOCK_SIZE = 32 # inference is always one image at a time.
        shuffled_mult_out = matmul_ERR(shuffled_weights, 
                                       op_in,
                                       BLOCK_HEIGHT, 
                                       BLOCK_WIDTH, 
                                       BATCH_BLOCK_SIZE, 
                                       ERR_PROFILE=error_profile_op,
                                       ERR_PARAM_TF=ERR_PARAM_TF)
    else:
        shuffled_mult_out = tf.linalg.matmul(shuffled_weights, op_in)

    ## was the weight matrix shuffled
    if shuffle_order_op is not None:
        # unshuffle mult_out
        indices = tf.expand_dims(shuffle_order_op, axis=1)
        updates = tf.range(tf.size(indices))
        shape = shuffle_order_op.shape
        scatter = tf.scatter_nd(indices, updates, shape)
        op_mult_out = tf.gather(shuffled_mult_out, scatter)
    else:
        op_mult_out = shuffled_mult_out


    # Add bias
    op_bout = tf.add(op_mult_out, tf.expand_dims(op_biases,axis=1))
    # Softmax
    op_out = tf.nn.softmax(op_bout, axis=0)
    # Tranpose to standard order
    class_scores = tf.transpose(op_out, perm=[1,0])
    # Get predictions
    predictions = tf.math.argmax(class_scores, axis=1)
    ## count no. of wrong predicitons
    # return predictions
    return tf.math.reduce_sum(tf.cast(tf.math.not_equal(tf.cast(b_labels, dtype=tf.int64), predictions), dtype=tf.int64))
##################################################
def ff_fashion_cnn2_ERR(model,
                    error_profile_c0,
                    error_profile_c1,
                    error_profile_h0,
                    error_profile_op,
                    ERR_PARAM,
                    shuffle_order_c0,
                    shuffle_order_c1,
                    shuffle_order_h0,
                    shuffle_order_op,
                    test_set,
                    batchsize):

    if ERR_PARAM is not None:
        ERR_PARAM_TF = tf.constant(ERR_PARAM)
    else:
        ERR_PARAM_TF = None
    # create dataset
    test_dataset = tf.data.Dataset.from_tensor_slices(test_set)
    # Dataset size should be a multiple of BATCH_SIZE
    test_dataset = test_dataset.batch(batchsize, drop_remainder=True)
    tot_images = test_dataset.cardinality() *batchsize
    misses = 0
    for step, (b_images, b_labels) in enumerate(test_dataset):
        b_misses = batch_fashion_cnn2_ERR( b_images,
                                        b_labels,
                                        model,
                                        error_profile_c0,
                                        error_profile_c1,
                                        error_profile_h0,
                                        error_profile_op,
                                        ERR_PARAM_TF,
                                        shuffle_order_c0,
                                        shuffle_order_c1,
                                        shuffle_order_h0,
                                        shuffle_order_op)

        misses = misses + b_misses
    return (tot_images-misses)/tot_images
##################################################
def eval_fashion_cnn2_ERR( model,
                        error_profile_c0,
                        error_profile_c1,
                        error_profile_h0,
                        error_profile_op,
                        ERR_PARAM,
                        shuffle_order_c0,
                        shuffle_order_c1,
                        shuffle_order_h0,
                        shuffle_order_op,
                        test_set):
    N_RUNS_PER_SHUFF_ORDER = 3
    BATCHSIZE = 128
    shuffle_result = []
    # Evaluate N_RUNS_PER_SHUFF_ORDER times for each shuffle order
    for i in range(N_RUNS_PER_SHUFF_ORDER):
        accuracy = ff_fashion_cnn2_ERR(model,
                                        error_profile_c0,
                                        error_profile_c1,
                                        error_profile_h0,
                                        error_profile_op,
                                        ERR_PARAM,
                                        shuffle_order_c0,
                                        shuffle_order_c1,
                                        shuffle_order_h0,
                                        shuffle_order_op,
                                        test_set,
                                        BATCHSIZE).numpy()
        shuffle_result.append(accuracy)
    return (np.mean(shuffle_result), 
            np.std(shuffle_result))
##################################################
    
    