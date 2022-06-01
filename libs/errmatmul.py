import tensorflow as tf
import numpy as np
import warnings

# CONSTANTS
N_THREADS_PER_BLOCK = 32

###############################################################################################################################
@tf.function
def matmul_ERRexpbitflips(tf_mat_A, 
                       tf_mat_B,
                       BLOCK_HEIGHT, 
                       BLOCK_WIDTH, 
                       BATCH_BLOCK_SIZE, 
                       ERR_PROFILE=None,
                       ERR_PARAM_TF=None,): # 0: flip-to-zero, 1: flip-to-1, -1: flip-bit, 2: TF32 mantissa truncation, 3: BF16 mantissa truncation
    
    """ 
        Bits of the exponent in FP32 representation are flipped for ERR 0,1,-1
        LSBs of the mantissa in FP32 are truncated for ERR 2, 3
        
        Assuming the shape dims of tf_mat_A are perfect multiples of BLOCK_HEIGHT and BLOCK_WIDTH.
        BLOCK_HEIGHT <= no. of rows of tf_mat_A
        BLOCK_WIDTH <= no. of cols of tf_mat_A
        
        Assuming the shape dims of tf_mat_B are perfect multiples of BLOCK_WIDTH and BATCH_BLOCK_SIZE.
        BATCH_BLOCK_SIZE <= no. of cols of tf_mat_B
    """
    # Add assertion checks for all these above variables
    tot_rows, tot_cols = tf_mat_A.shape
    gridydim = int(tot_rows/BLOCK_HEIGHT) 
    gridxdim = int(tot_cols/BLOCK_WIDTH)
    _ , BATCH_SIZE = tf_mat_B.shape
    gridbatchdim = int(BATCH_SIZE/BATCH_BLOCK_SIZE)
    
    # if BATCH_SIZE is not a multiple of a BATCH_BLOCK_SIZE
    # gridbatchdim = 1 if BATCH_SIZE < BATCH_BLOCK_SIZE else int(BATCH_SIZE/BATCH_BLOCK_SIZE)
    # BATCH_BLOCK_SIZE = int(BATCH_SIZE/gridbatchdim)
    
    
    tf_mat_A_tiles = tf.transpose(tf.reshape(tf_mat_A, [gridydim, BLOCK_HEIGHT, -1, BLOCK_WIDTH]), perm=[0,2,1,3])
    tf_mat_B_tiles = tf.transpose(tf.reshape(tf_mat_B, [gridxdim,BLOCK_WIDTH,-1, BATCH_BLOCK_SIZE]), perm=[0,2,1,3])
    tf_ein_result = tf.einsum('lmij,mnjk->lnmik',tf_mat_A_tiles, tf_mat_B_tiles)
    
    
    if ERR_PROFILE is not None:
        # assert ERR_PARAM_TF is not None, "ERR_PARAM_TF requires a value."
        
        # GENERATE ERROR MASK from given ERR_PROFILE
        ## extract the error profile depending on
        ### how many GPU blocks were used
        ### how many threads per block are used
        BLOCKS_USED = gridydim*gridxdim
        # ERR_PROFILE_USED = tf.constant(ERR_PROFILE[0:BLOCKS_USED, 0:BLOCK_HEIGHT])
        ERR_PROFILE_USED = ERR_PROFILE[0:BLOCKS_USED, 0:BLOCK_HEIGHT]
        ERR_PROFILE_LAYOUT = tf.reshape(ERR_PROFILE_USED, shape=(gridydim, gridxdim, BLOCK_HEIGHT))

        ## Sample from given layout to get error_mask_layout
        ## error_mask_layout represents which values of the tensor will be corrupted
        bernoulli_sample = tf.keras.backend.random_bernoulli(shape=(gridbatchdim,BATCH_BLOCK_SIZE, *ERR_PROFILE_LAYOUT.shape),
                                                             p=ERR_PROFILE_LAYOUT) # 1 -> error present, 0 -> no error
        error_mask = tf.ones_like(bernoulli_sample) - bernoulli_sample # 0 -> error present, 1 -> no error

        ## Reshape the error_mask so that it is compatible with the layout of the einsum output
        ## refer to shaping_error_to_einsum_output.ipynb notebook for details about the transpose logic
        error_mask_layout = tf.transpose(error_mask,perm=[2,0,3,4,1]) # 0 -> error present, 1 -> no error
        inv_error_mask_layout = tf.ones_like(error_mask_layout) - error_mask_layout # 1 -> error present, 0 -> no error
        
        # DEFINE THE CORRUPTION
        ## Define which bit of the exponent is being corrupted
        ## Bits 0 to 7 of exponent may be corrupted
        exponent_size = 8
        fraction_size = 23
        corrupt_bitpos = tf.random.uniform(shape=tf.shape(inv_error_mask_layout), minval=0, maxval=exponent_size-1, dtype=tf.int32)
        ## Generate the corruption_mask using the values of corrupt_bitpos
        corrupt_bitpos_mask =  tf.cast(tf.bitwise.left_shift(1, fraction_size+corrupt_bitpos), dtype=tf.int32)
        ## Filter out only those values where error_mask is TRUE
        corrupt_mask = tf.cast(inv_error_mask_layout, dtype=tf.int32) * tf.cast(corrupt_bitpos_mask, dtype=tf.int32)
        
        # Masks to truncate last bits of mantissa
        all_ones = tf.ones_like(error_mask_layout) * 0xFFFF_FFFF
        all_ones = tf.cast(all_ones, dtype=tf.int32)
        # TF32 [1+8+10]
        truncate_size_tf32 = 13 # truncate (zero out) the last 13 bits of FP32 number. Mantissa size is 23-13 = 10
        truncate_mask_tf32 = tf.cast(inv_error_mask_layout, dtype=tf.int32) * tf.cast(tf.bitwise.left_shift(all_ones, truncate_size_tf32), dtype=tf.int32)
        # BF16 [1+8+7]
        truncate_size_bf16 = 16 # truncate (zero out) the last 16 bits of FP32 number. Mantissa size is 23-16 = 7
        truncate_mask_bf16 = tf.cast(tf.bitwise.left_shift(all_ones, truncate_size_bf16), dtype=tf.int32)
                
        # APPLY ERROR MASK TO THE einsum RESULT
        bitcast_to_int32 = tf.bitcast(tf_ein_result, tf.int32)
        if ERR_PARAM_TF == -1: # flip bit
            flipbits = tf.bitwise.bitwise_xor(bitcast_to_int32, corrupt_mask)
        elif ERR_PARAM_TF == 1: # set to 1
            flipbits = tf.bitwise.bitwise_or(bitcast_to_int32, corrupt_mask)
        elif ERR_PARAM_TF == 0: # set to 0
            flipbits = tf.bitwise.bitwise_and(bitcast_to_int32, tf.bitwise.invert(corrupt_mask))
        elif ERR_PARAM_TF == 2: # truncate to TF32
            dont_truncate = bitcast_to_int32 * tf.cast(error_mask_layout,     dtype=tf.int32) # 0 -> error present, 1 -> no error
            to_truncate   = bitcast_to_int32 * tf.cast(inv_error_mask_layout, dtype=tf.int32) # 1 -> error present, 0 -> no error
            truncated     = tf.bitwise.bitwise_and(to_truncate, truncate_mask_tf32)
            flipbits      = dont_truncate + truncated
        elif ERR_PARAM_TF == 3: # truncate to BF16
            dont_truncate = bitcast_to_int32 * tf.cast(error_mask_layout,     dtype=tf.int32) # 0 -> error present, 1 -> no error
            to_truncate   = bitcast_to_int32 * tf.cast(inv_error_mask_layout, dtype=tf.int32) # 1 -> error present, 0 -> no error
            truncated     = tf.bitwise.bitwise_and(to_truncate, truncate_mask_bf16)
            flipbits      = dont_truncate + truncated
        else:
            # warnings.warn('ERR_PARAM_TF should have value 0, 1, or -1. Continuing without flipping bits')
            flipbits = bitcast_to_int32
            
        bitcast_to_float32 = tf.bitcast(flipbits, tf.float32)
        corrupted_result = bitcast_to_float32
    else:
        corrupted_result = tf_ein_result

    reduce_result = tf.math.reduce_sum(corrupted_result, axis=2) 
    tf_rshp = tf.reshape(reduce_result, shape=(gridydim,gridbatchdim,BLOCK_HEIGHT,BATCH_BLOCK_SIZE))
    tf_rshp_tr = tf.transpose(tf_rshp, perm=[0,2,1,3])
    tf_final_result = tf.reshape(tf_rshp_tr, shape=(gridydim*BLOCK_HEIGHT,gridbatchdim*BATCH_BLOCK_SIZE))
    return tf_final_result
###############################################################################################################################
