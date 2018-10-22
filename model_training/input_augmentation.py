import tensorflow as tf
import model_layers

### Augmentations
def augment_input_sequences(cur_input_seq, cur_rois):
    _, T, H, W, C = cur_input_seq.shape.as_list()
    ##### AUGMENTATION
    top, left, bottom, right = cur_rois[:,0], cur_rois[:,1], cur_rois[:,2], cur_rois[:,3]

    ### flipping image and coords left-right
    flip_mux = tf.random_uniform([], minval=0, maxval=1) > 0.5

    #image
    flipped_input = cur_input_seq[:,:,:,::-1,:]
    input_seq = tf.cond(flip_mux, lambda:cur_input_seq, lambda: flipped_input)

    #coords
    flipped_left = 1.0 - right
    flipped_right = 1.0 - left
    new_left  = tf.cond(flip_mux, lambda:left, lambda: flipped_left)
    new_right = tf.cond(flip_mux, lambda:right, lambda: flipped_right)
    augmented_rois = tf.stack([top, new_left, bottom, new_right], axis=1)

    # cur_input_seq = tf.cond(self.is_training, lambda:input_seq, lambda: cur_input_seq)
    # cur_rois = tf.cond(self.is_training, lambda:augmented_rois, lambda: cur_rois)
    cur_input_seq = input_seq
    cur_rois = augmented_rois

    ### cropping and resizing
    top, left, bottom, right = cur_rois[:,0], cur_rois[:,1], cur_rois[:,2], cur_rois[:,3]
    B = tf.shape(cur_input_seq)[0]

    offset_max = 0.10
    left_offset   = tf.random_uniform([], minval=-offset_max, maxval=offset_max)
    right_offset  = tf.random_uniform([], minval=-offset_max, maxval=offset_max)
    top_offset    = tf.random_uniform([], minval=-offset_max, maxval=offset_max)
    bottom_offset = tf.random_uniform([], minval=-offset_max, maxval=offset_max)

    left_crop = left_offset
    right_crop = 1.0 - right_offset
    top_crop = top_offset
    bottom_crop = 1.0 - bottom_offset

    # update box_coords
    top, left, bottom, right = cur_rois[:,0], cur_rois[:,1], cur_rois[:,2], cur_rois[:,3]
    updated_left   = (left - left_offset) / (1.0 - left_offset - right_offset)
    updated_right  = (right - left_offset) / (1.0 - left_offset - right_offset)

    updated_top    = (top - top_offset) / (1.0 - top_offset - bottom_offset)
    updated_bottom = (bottom - top_offset) / (1.0 - top_offset - bottom_offset)

    augmented_rois = tf.stack([updated_top, updated_left, updated_bottom, updated_right], axis=1)

    # crop images
    tiled_left = tf.expand_dims(left_crop, axis=0)
    tiled_left = tf.tile(tiled_left, [B])

    tiled_right = tf.expand_dims(right_crop, axis=0)
    tiled_right = tf.tile(tiled_right, [B])

    tiled_top = tf.expand_dims(top_crop, axis=0)
    tiled_top = tf.tile(tiled_top, [B])

    tiled_bottom = tf.expand_dims(bottom_crop, axis=0)
    tiled_bottom = tf.tile(tiled_bottom, [B])

    new_boxes = tf.stack([tiled_top, tiled_left, tiled_bottom, tiled_right], axis=1)

    cropped_sequence = model_layers.temporal_roi_cropping(cur_input_seq, new_boxes, tf.range(B), [H, W])

    return cropped_sequence, augmented_rois

def augment_box_coords(cur_rois):
    # quick augmentation on roi coords
    # coord_aug_val = 0.10
    #coord_aug_val = 0.00
    #box_coord_shifts = tf.random_uniform([4], minval= -coord_aug_val, maxval= +coord_aug_val, name='ROI_Augmentation')

    #coord_delta = tf.cond(self.is_training, lambda:box_coord_shifts, lambda: tf.zeros([4]))

    #shifted_rois = cur_rois + coord_delta

    ## BBox area augmentation
    ## [top, left, bottom, right]
    R = tf.shape(cur_rois)[0] # no of rois

    top, left, bottom, right = cur_rois[:,0], cur_rois[:,1], cur_rois[:,2], cur_rois[:,3]
    height = bottom - top
    width = right - left
    y_center = (bottom + top) / 2
    x_center = (right + left) / 2
    area = height * width
    ratio = height / width

    # augment area
    # area_aug_mult = tf.random_uniform([R], minval=0.50, maxval=2.00, name='AreaAugmentationMult')
    # reduce or increase area with same prob
    area_aug_mux = tf.random_uniform([R], minval=-1.0, maxval=1.0, name='AreaAugmentationMult')
    min_area_mult = 0.25
    max_area_mult = 2.00
    area_reducers = -area_aug_mux * (1.-min_area_mult) + min_area_mult
    area_reducers = area_reducers * tf.cast(area_aug_mux < 0.0, tf.float32) # only take mux < 0
    area_increasers = area_aug_mux * (max_area_mult-1.) + 1.
    area_increasers = area_increasers * tf.cast(area_aug_mux > 0.0, tf.float32) # only take mux>0
    area_aug_mult = area_reducers + area_increasers
    area_augmented = area * area_aug_mult
    
    # augment ratio
    ratio_aug_mult = tf.random_uniform([R], minval=0.75, maxval=1.25, name='RatioAugmentationMult')
    ratio_augmented = ratio * ratio_aug_mult

    # augment center
    ycenter_aug_mult = tf.random_uniform([R], minval=-0.15, maxval=0.15, name='YCenterAugmentationMult')
    y_center_augmented = y_center + ycenter_aug_mult * height

    xcenter_aug_mult = tf.random_uniform([R], minval=-0.15, maxval=0.15, name='XCenterAugmentationMult')
    x_center_augmented = x_center + xcenter_aug_mult * width

    # calculate the augmented bbox coords
    height_aug = tf.sqrt(area_augmented*ratio_augmented)
    width_aug = height_aug / ratio_augmented

    top_aug = y_center_augmented - height_aug / 2.0
    bottom_aug = y_center_augmented + height_aug / 2.0

    left_aug = x_center_augmented - width_aug / 2.0
    right_aug = x_center_augmented + width_aug / 2.0

    # # Randomly flip left right coordinates. crop_and_resize allows left>right in which case flips the image
    # flip_mux = tf.random_uniform([R], minval=0, maxval=1)
    # choose_left_from_left = tf.cast(flip_mux < 0.5, tf.float32)
    # choose_left_from_right = 1. - choose_left_from_left
    # left_right = tf.stack([left_aug, right_aug], axis=1)
    # choose_left_mtx =  tf.stack([choose_left_from_left, choose_left_from_right], axis=1)
    # flipped_left = tf.reduce_sum(left_right * choose_left_mtx, axis=1)
    # flipped_right = tf.reduce_sum(left_right * (1-choose_left_mtx), axis=1)

    ### flipped_left = tf.cond(flip_switch<0.5, lambda: left_aug, lambda:right_aug)
    ### flipped_right = tf.cond(flip_switch<0.5, lambda: right_aug, lambda:left_aug)
    
    # augmented_rois = tf.stack([top_aug, flipped_left, bottom_aug, flipped_right], axis=1)
    augmented_rois = tf.stack([top_aug, left_aug, bottom_aug, right_aug], axis=1)

    return augmented_rois