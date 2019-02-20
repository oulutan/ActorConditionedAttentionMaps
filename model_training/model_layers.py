import tensorflow as tf
import i3d as i3d_model

BOX_CROP_SIZE = [10,10]



### Layers after initial I3D Head
def choose_roi_architecture(architecture_str, features, shifted_rois, cur_b_idx, is_training):
    if architecture_str == 'i3d_tail':
        box_features =  temporal_roi_cropping(features, shifted_rois, cur_b_idx, BOX_CROP_SIZE)
        class_feats = i3d_tail_model(box_features, is_training)
    elif architecture_str == 'basic_model':
        box_features =  temporal_roi_cropping(features, shifted_rois, cur_b_idx, BOX_CROP_SIZE)
        class_feats = basic_model(box_features)
    elif architecture_str == 'non_local_v1':
        box_features =  temporal_roi_cropping(features, shifted_rois, cur_b_idx, BOX_CROP_SIZE)
        class_feats =  non_local_ROI_model(box_features, features, cur_b_idx, is_training)
    elif  architecture_str == 'non_local_attn':
        box_features =  temporal_roi_cropping(features, shifted_rois, cur_b_idx, BOX_CROP_SIZE)
        class_feats =  non_local_ROI_feat_attention_model(box_features, features, cur_b_idx)
    elif  architecture_str == 'soft_attn':
        class_feats =  soft_roi_attention_model(features, shifted_rois, cur_b_idx, is_training)
    elif  architecture_str == 'acrn_roi':
        class_feats =  acrn_roi_model(features, shifted_rois, cur_b_idx, BOX_CROP_SIZE)
    elif  architecture_str == 'single_attn':
        class_feats =  single_soft_roi_attention_model(features, shifted_rois, cur_b_idx, BOX_CROP_SIZE)
    elif  architecture_str == 'non_local_v2':
        class_feats =  non_local_ROI_model_v2(features, shifted_rois, cur_b_idx, BOX_CROP_SIZE)
    elif  architecture_str == 'non_local_v3':
        class_feats =  non_local_ROI_model_v3(features, shifted_rois, cur_b_idx, BOX_CROP_SIZE)
    elif  architecture_str == 'multi_non_local':
        class_feats =  multi_non_local_block_roi(features, shifted_rois, cur_b_idx, BOX_CROP_SIZE)
    else:
        print('Architecture not implemented!')
        raise NotImplementedError

    return class_feats

### Layers
def basic_model(roi_box_features):
    # basic model, takes the input feature and averages across temporal dim
    # temporal_len = roi_box_features.shape[1]
    B, temporal_len, H, W, C = roi_box_features.shape
    avg_features = tf.nn.avg_pool3d(      roi_box_features,
                                            ksize=[1, temporal_len, 1, 1, 1],
                                            strides=[1, temporal_len, 1, 1, 1],
                                            padding='VALID',
                                            name='TemporalPooling')
    # classification
    class_feats = tf.layers.flatten(avg_features)

    return class_feats

def basic_model_pooled(roi_box_features):
    # basic model, takes the input feature and averages across temporal dim
    # temporal_len = roi_box_features.shape[1]
    B, temporal_len, H, W, C = roi_box_features.shape
    avg_features = tf.nn.avg_pool3d(      roi_box_features,
    #avg_features = tf.nn.max_pool3d(      roi_box_features,
                                            ksize=[1, temporal_len, H, W, 1],
                                            strides=[1, temporal_len, H, W, 1],
                                            padding='VALID',
                                            name='TemporalPooling')
    # classification
    class_feats = tf.layers.flatten(avg_features)

    return class_feats

def i3d_tail_model(roi_box_features, is_training):
    # I3D continued after mixed4e
    with tf.variable_scope('Tail_I3D'):
        tail_end_point = 'Mixed_5c'
        # tail_end_point = 'Mixed_4f'
        final_i3d_feat, end_points = i3d_model.i3d_tail(roi_box_features, is_training, tail_end_point)
        # final_i3d_feat = end_points[tail_end_point]
        tf.add_to_collection('final_i3d_feats', final_i3d_feat)
        
        
        # flat_feats = tf.layers.flatten(final_i3d_feat)
        flat_feats = basic_model_pooled(final_i3d_feat)
        # import pdb;pdb.set_trace()
        pass
        

    return flat_feats

def non_local_ROI_model(roi_box_features, context_features, cur_b_idx, is_training):
    '''
    roi_box_features: bounding box features extracted on detected people
    context_features: main feature map extracted from full frame
    cur_b_idx: Batch - Roi mapping ex: 5 rois, 3 batch segments then an example [0,0,1,1,2]
    '''
    with tf.variable_scope('Non_Local_Block'):
        _, Tr, Hr, Wr, Cr = roi_box_features.shape.as_list()
        R = tf.shape(roi_box_features)[0]
        _, Tc, Hc, Wc, Cc = context_features.shape.as_list()
        B = tf.shape(context_features)[0]


        feature_map_channel = Cr / 2

        roi_embedding = tf.layers.conv3d(roi_box_features, filters=feature_map_channel, kernel_size=[1,1,1], padding='SAME', activation=tf.nn.relu, name='RoiEmbedding')
        context_embedding = tf.layers.conv3d(context_features, filters=feature_map_channel, kernel_size=[1,1,1], padding='SAME', activation=tf.nn.relu, name='ContextEmbedding')

        context_response = tf.layers.conv3d(context_features, filters=feature_map_channel, kernel_size=[1,1,1], padding='SAME', activation=tf.nn.relu, name='ContextRepresentation')

        # Number of rois(R) is larger than number of batches B as from each segment we extract multiple rois
        # we need to gather batches such that rois are assigned to correct context features that they were extracted from
        context_embedding_gathered = tf.gather(context_embedding, cur_b_idx, axis=0, name='ContextEmbGather')
        context_response_gathered = tf.gather(context_response, cur_b_idx, axis=0, name='ContextResGather')
        # now they have R as first dimensions

        # reshape so that we can use matrix multiplication to calculate attention mapping
        roi_emb_reshaped = tf.reshape(roi_embedding, shape=[R, Tr*Hr*Wr, feature_map_channel])
        context_emb_reshaped = tf.reshape(context_embedding_gathered, shape=[R, Tc*Hc*Wc, feature_map_channel])
        context_res_reshaped = tf.reshape(context_response_gathered, shape=[R, Tc*Hc*Wc, feature_map_channel])

        emb_mtx = tf.matmul(roi_emb_reshaped, context_emb_reshaped, transpose_b=True) # [R,Tr*Hr*Wr, Tc*Hc*Wc]
        emb_mtx = emb_mtx / tf.sqrt(tf.cast(feature_map_channel, tf.float32)) # normalization of rand variables

        embedding_attention = tf.nn.softmax(emb_mtx, name='EmbeddingNormalization')

        attention_response = tf.matmul(embedding_attention, context_res_reshaped) # [R, Tr*Hr*Wr, feature_map_channel]

        attention_response_org_shape = tf.reshape(attention_response, [R, Tr, Hr, Wr, feature_map_channel])

        # blow it up to original feature dimension
        non_local_feature = tf.layers.conv3d(attention_response_org_shape, filters=Cr, kernel_size=[1,1,1], padding='SAME', activation=tf.nn.relu, name='NonLocalFeature')

        # Residual connection
        residual_feature = roi_box_features + non_local_feature

    # i3d_tail_feats = i3d_tail_model(residual_feature, is_training)

    # return i3d_tail_feats
    # i3d_tail_feats = self.i3d_tail_model(residual_feature)
    # return i3d_tail_feats
    with tf.variable_scope('Tail_I3D'):
        tail_end_point = 'Mixed_5c'
        # tail_end_point = 'Mixed_4f'
        final_i3d_feat, end_points = i3d_model.i3d_tail(residual_feature, is_training, tail_end_point)
    
    # classification
    class_feats = basic_model(final_i3d_feat)
    return class_feats

# def non_local_ROI_feat_attention_model(self, roi_box_features, context_features, cur_b_idx):
#     '''
#     roi_box_features: bounding box features extracted on detected people
#     context_features: main feature map extracted from full frame
#     cur_b_idx: Batch - Roi mapping ex: 5 rois, 3 batch segments then an example [0,0,1,1,2]
#     '''
#     with tf.variable_scope('Non_Local_Block'):
#         _, Tr, Hr, Wr, Cr = roi_box_features.shape.as_list()
#         R = tf.shape(roi_box_features)[0]
#         _, Tc, Hc, Wc, Cc = context_features.shape.as_list()
#         B = tf.shape(context_features)[0]


#         feature_map_channel = Cr / 64

#         roi_embedding = tf.layers.conv3d(roi_box_features, filters=feature_map_channel, kernel_size=[1,1,1], padding='SAME', activation=tf.nn.relu, name='RoiEmbedding')
#         # roi_embedding = roi_box_features

#         context_embedding = tf.layers.conv3d(context_features, filters=feature_map_channel, kernel_size=[1,1,1], padding='SAME', activation=tf.nn.relu, name='ContextEmbedding')

#         context_response = tf.layers.conv3d(context_features, filters=feature_map_channel, kernel_size=[1,1,1], padding='SAME', activation=tf.nn.relu, name='ContextRepresentation')

#         # Number of rois(R) is larger than number of batches B as from each segment we extract multiple rois
#         # we need to gather batches such that rois are assigned to correct context features that they were extracted from
#         context_embedding_gathered = tf.gather(context_embedding, cur_b_idx, axis=0, name='ContextEmbGather')
#         context_response_gathered = tf.gather(context_response, cur_b_idx, axis=0, name='ContextResGather')
#         # now they have R as first dimensions

#         # reshape so that we can use matrix multiplication to calculate attention mapping
#         roi_emb_reshaped = tf.reshape(roi_embedding, shape=[R, Tr*Hr*Wr, feature_map_channel, 1])
#         roi_emb_permuted = tf.transpose(roi_emb_reshaped, perm=[0,2,1,3]) # [R, feature_map_channel, Tr*Hr*Wr, 1]
        
#         context_emb_reshaped = tf.reshape(context_embedding_gathered, shape=[R, Tc*Hc*Wc, feature_map_channel, 1])
#         context_emb_permuted = tf.transpose(context_emb_reshaped, perm=[0,2,1,3]) # [R, feature_map_channel, Tc*Hc*Wc, 1]

#         context_res_reshaped = tf.reshape(context_response_gathered, shape=[R, Tc*Hc*Wc, feature_map_channel])
#         context_res_permuted = tf.transpose(context_res_reshaped, perm=[0,2,1]) # [R, feature_map_channel, Tc*Hc*Wc]

#         emb_mtx = tf.matmul(roi_emb_permuted, context_emb_permuted, transpose_b=True) # [R, feature_map_channel,Tr*Hr*Wr, Tc*Hc*Wc]
#         # emb_mtx = emb_mtx / tf.sqrt(tf.cast(feature_map_channel, tf.float32)) # normalization of rand variables

#         embedding_mtx_permuted = tf.transpose(emb_mtx, [0, 2, 1, 3]) # [R,Tr*Hr*Wr, feature_map_channel, Tc*Hc*Wc]

#         embedding_attention = tf.nn.softmax(embedding_mtx_permuted, name='EmbeddingNormalization') # get the weights

#         context_res_expanded = tf.expand_dims(context_res_permuted, axis=1)
#         context_res_tiled = tf.tile(context_res_expanded, [1,Tr*Hr*Wr,1,1]) # [R,Tr*Hr*Wr, feature_map_channel, Tc*Hc*Wc]

#         attention_response = tf.multiply(embedding_attention, context_res_tiled) # [R,Tr*Hr*Wr, feature_map_channel, Tc*Hc*Wc]

#         attention_response_reduced = tf.reduce_sum(attention_response, axis=3) # this final sum gives the weighted sum of context_responses

#         attention_response_org_shape = tf.reshape(attention_response_reduced, [R, Tr, Hr, Wr, feature_map_channel])

#         # blow it up to original feature dimension
#         non_local_feature = tf.layers.conv3d(attention_response_org_shape, filters=Cr, kernel_size=[1,1,1], padding='SAME', activation=tf.nn.relu, name='NonLocalFeature')

#         # Residual connection
#         residual_feature = roi_box_features + non_local_feature

#     i3d_tail_feats = self.i3d_tail_model(residual_feature)

#     return i3d_tail_feats

def soft_roi_attention_model(context_features, shifted_rois, cur_b_idx, is_training):
    with tf.variable_scope('Soft_Attention_Model'):
        _, Tc, Hc, Wc, Cc = context_features.shape.as_list()
        B = tf.shape(context_features)[0]
        feature_map_channel = Cc / 4

        roi_box_features = temporal_roi_cropping(context_features, shifted_rois, cur_b_idx, BOX_CROP_SIZE)
        R = tf.shape(shifted_rois)[0]

        flat_box_feats = basic_model_pooled(roi_box_features)
        roi_embedding = tf.layers.dense(flat_box_feats, 
                                        feature_map_channel, 
                                        activation=tf.nn.relu,
                                        kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01),
                                        name='RoiEmbedding')
        roi_embedding = tf.layers.dropout(inputs=roi_embedding, rate=0.5, training=is_training, name='RoI_Dropout')

        context_embedding = tf.layers.conv3d(context_features, 
                                            filters=feature_map_channel, 
                                            kernel_size=[1,1,1], 
                                            padding='SAME', 
                                            activation=tf.nn.relu, 
                                            kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01),
                                            name='ContextEmbedding')
        context_embedding =  tf.layers.dropout(inputs=context_embedding, rate=0.5, training=is_training, name='Context_Dropout')

        # roi_embedding = tf.layers.dropout(inputs=flat_box_feats, rate=0.5, training=is_training, name='RoI_Dropout')
        # context_embedding =  tf.layers.dropout(inputs=context_features, rate=0.5, training=is_training, name='Context_Dropout')
        
        # with tf.device('/cpu:0'):
        roi_expanded = tf.expand_dims(tf.expand_dims(tf.expand_dims(roi_embedding, axis=1), axis=1), axis=1) # R,512 -> R,1,1,1,512
        roi_tiled = tf.tile(roi_expanded, [1,Tc,Hc,Wc,1], 'RoiTiling')

        # multiply context_feats by no of rois so we can concatenate
        context_embedding_gathered = tf.gather(context_embedding, cur_b_idx, axis=0, name='ContextEmbGather')

        roi_context_feats = tf.concat([roi_tiled, context_embedding_gathered], 4, name='RoiContextConcat')

        relation_feats = tf.layers.conv3d(  roi_context_feats, 
                                            filters=Cc, 
                                            kernel_size=[1,1,1], 
                                            padding='SAME', 
                                            activation=None, 
                                            kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01),
                                            name='RelationFeats')
        
        attention_map = tf.nn.sigmoid(relation_feats,'AttentionMap') # use sigmoid so it represents a heatmap of attention
        # heatmap of attention
        tf.add_to_collection('attention_map', attention_map) # for attn map generation
        
        # with tf.device('/cpu:0'):
        # Multiply attention map with context features. Now this new feature represents the roi
        gathered_context = tf.gather(context_features, cur_b_idx, axis=0, name='ContextGather')
        tf.add_to_collection('feature_activations', gathered_context) # for attn map generation
        soft_attention_feats = tf.multiply(attention_map, gathered_context)
    

    class_feats = i3d_tail_model(soft_attention_feats, is_training)
    return class_feats
    # with tf.variable_scope('Tail_I3D'):
    #     tail_end_point = 'Mixed_5c'
    #     # tail_end_point = 'Mixed_4f'
    #     final_i3d_feat, end_points = i3d_model.i3d_tail(soft_attention_feats, is_training, tail_end_point)
    # 
    # temporal_len = final_i3d_feat.shape[1]
    # avg_features = tf.nn.avg_pool3d(      final_i3d_feat,
    #                                         ksize=[1, temporal_len, 3, 3, 1],
    #                                         strides=[1, temporal_len, 3, 3, 1],
    #                                         padding='SAME',
    #                                         name='TemporalPooling')
    # # classification
    # class_feats = tf.layers.flatten(avg_features)

    # return class_feats

def acrn_roi_model(context_features, shifted_rois, cur_b_idx, is_training):
    with tf.variable_scope('Soft_Attention_Model'):
        _, Tc, Hc, Wc, Cc = context_features.shape.as_list()
        B = tf.shape(context_features)[0]
        feature_map_channel = Cc / 4

        roi_box_features = temporal_roi_cropping(context_features, shifted_rois, cur_b_idx, BOX_CROP_SIZE)
        R = tf.shape(shifted_rois)[0]

        flat_box_feats = basic_model(roi_box_features)
        roi_embedding = tf.layers.dense(flat_box_feats, 
                                        feature_map_channel, 
                                        activation=tf.nn.relu,
                                        kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01),
                                        name='RoiEmbedding')

        context_embedding = tf.layers.conv3d(context_features, 
                                            filters=feature_map_channel, 
                                            kernel_size=[1,1,1], 
                                            padding='SAME', 
                                            activation=tf.nn.relu, 
                                            name='ContextEmbedding')
        
        # with tf.device('/cpu:0'):
        roi_expanded = tf.expand_dims(tf.expand_dims(tf.expand_dims(roi_embedding, axis=1), axis=1), axis=1) # R,512 -> R,1,1,1,512
        roi_tiled = tf.tile(roi_expanded, [1,Tc,Hc,Wc,1], 'RoiTiling')

        # multiply context_feats by no of rois so we can concatenate
        context_embedding_gathered = tf.gather(context_embedding, cur_b_idx, axis=0, name='ContextEmbGather')

        roi_context_feats = tf.concat([roi_tiled, context_embedding_gathered], 4, name='RoiContextConcat')

        relation_feats = tf.layers.conv3d(  roi_context_feats, 
                                            filters=Cc, 
                                            kernel_size=[1,1,1], 
                                            padding='SAME', 
                                            activation=tf.nn.relu, 
                                            name='RelationFeats')
        
        #attention_map = tf.nn.sigmoid(relation_feats,'AttentionMap') # use sigmoid so it represents a heatmap of attention
        # heatmap of attention
        
        # with tf.device('/cpu:0'):
        # Multiply attention map with context features. Now this new feature represents the roi
        #gathered_context = tf.gather(relation_feats, cur_b_idx, axis=0, name='ContextGather')
        tf.add_to_collection('feature_activations', relation_feats) # for attn map generation
        #soft_attention_feats = tf.multiply(attention_map, gathered_context)
    

    class_feats = i3d_tail_model(relation_feats, is_training)
    return class_feats



def single_soft_roi_attention_model(context_features, shifted_rois, cur_b_idx, is_training):
    with tf.variable_scope('Soft_Attention_Model'):
        _, Tc, Hc, Wc, Cc = context_features.shape.as_list()
        B = tf.shape(context_features)[0]
        feature_map_channel = Cc / 4

        roi_box_features = temporal_roi_cropping(context_features, shifted_rois, cur_b_idx, BOX_CROP_SIZE)
        R = tf.shape(shifted_rois)[0]

        flat_box_feats = basic_model(roi_box_features)
        roi_embedding = tf.layers.dense(flat_box_feats, 
                                        feature_map_channel, 
                                        activation=tf.nn.relu,
                                        kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01),
                                        name='RoiEmbedding')

        context_embedding = tf.layers.conv3d(context_features, 
                                            filters=feature_map_channel, 
                                            kernel_size=[1,1,1], 
                                            padding='SAME', 
                                            activation=tf.nn.relu, 
                                            name='ContextEmbedding')
        
        # with tf.device('/cpu:0'):
        roi_expanded = tf.expand_dims(tf.expand_dims(tf.expand_dims(roi_embedding, axis=1), axis=1), axis=1) # R,512 -> R,1,1,1,512
        roi_tiled = tf.tile(roi_expanded, [1,Tc,Hc,Wc,1], 'RoiTiling')

        # multiply context_feats by no of rois so we can concatenate
        context_embedding_gathered = tf.gather(context_embedding, cur_b_idx, axis=0, name='ContextEmbGather')

        roi_context_feats = tf.concat([roi_tiled, context_embedding_gathered], 4, name='RoiContextConcat')

        relation_feats = tf.layers.conv3d(  roi_context_feats, 
                                            filters=1, 
                                            kernel_size=[1,1,1], 
                                            padding='SAME', 
                                            activation=None, 
                                            name='RelationFeats')
        
        attention_map = tf.nn.sigmoid(relation_feats,'AttentionMap') # use sigmoid so it represents a heatmap of attention
        # heatmap of attention
        tf.add_to_collection('attention_map', attention_map) # for attn map generation
        
        # with tf.device('/cpu:0'):
        # Multiply attention map with context features. Now this new feature represents the roi
        gathered_context = tf.gather(context_features, cur_b_idx, axis=0, name='ContextGather')
        tf.add_to_collection('feature_activations', gathered_context) # for attn map generation
        soft_attention_feats = tf.multiply(attention_map, gathered_context)

    with tf.variable_scope('Tail_I3D'):
        tail_end_point = 'Mixed_5c'
        # tail_end_point = 'Mixed_4f'
        final_i3d_feat, end_points = i3d_model.i3d_tail(soft_attention_feats, is_training, tail_end_point)
    
    tf.add_to_collection('final_i3d_feats', final_i3d_feat)
    temporal_len = final_i3d_feat.shape[1]
    avg_features = tf.nn.avg_pool3d(      final_i3d_feat,
                                            ksize=[1, temporal_len, 3, 3, 1],
                                            strides=[1, temporal_len, 3, 3, 1],
                                            padding='SAME',
                                            name='TemporalPooling')
    # classification
    class_feats = tf.layers.flatten(avg_features)

    return class_feats

# def non_local_ROI_model_v2(self, context_features, shifted_rois, cur_b_idx, BOX_CROP_SIZE):
#     '''
#     roi_box_features: bounding box features extracted on detected people
#     context_features: main feature map extracted from full frame
#     cur_b_idx: Batch - Roi mapping ex: 5 rois, 3 batch segments then an example [0,0,1,1,2]
#     '''
#     with tf.variable_scope('Non_Local_Block'):
#         # _, Tr, Hr, Wr, Cr = roi_box_features.shape.as_list()
#         # R = tf.shape(roi_box_features)[0]
#         _, Tc, Hc, Wc, Cc = context_features.shape.as_list()
#         B = tf.shape(context_features)[0]

#         feature_map_channel = Cc / 2

#         roi_embedding = tf.layers.conv3d(context_features, filters=feature_map_channel, kernel_size=[1,1,1], padding='SAME', activation=tf.nn.relu, name='RoiEmbedding')
#         context_embedding = tf.layers.conv3d(context_features, filters=feature_map_channel, kernel_size=[1,1,1], padding='SAME', activation=tf.nn.relu, name='ContextEmbedding')

#         context_response = tf.layers.conv3d(context_features, filters=feature_map_channel, kernel_size=[1,1,1], padding='SAME', activation=tf.nn.relu, name='ContextRepresentation')


#         # reshape so that we can use matrix multiplication to calculate attention mapping
#         roi_emb_reshaped = tf.reshape(roi_embedding, shape=[B, Tc*Hc*Wc, feature_map_channel])
#         context_emb_reshaped = tf.reshape(context_embedding, shape=[B, Tc*Hc*Wc, feature_map_channel])
#         context_res_reshaped = tf.reshape(context_response, shape=[B, Tc*Hc*Wc, feature_map_channel])

#         emb_mtx = tf.matmul(roi_emb_reshaped, context_emb_reshaped, transpose_b=True) # [R,Tr*Hr*Wr, Tc*Hc*Wc]

#         embedding_attention = tf.nn.softmax(emb_mtx, name='EmbeddingNormalization')

#         attention_response = tf.matmul(embedding_attention, context_res_reshaped) # [R, Tr*Hr*Wr, feature_map_channel]

#         attention_response_org_shape = tf.reshape(attention_response, [B, Tc, Hc, Wc, feature_map_channel])

#         # blow it up to original feature dimension
#         non_local_feature = tf.layers.conv3d(attention_response_org_shape, filters=Cc, kernel_size=[1,1,1], padding='SAME', activation=tf.nn.relu, name='NonLocalFeature')

#         # Residual connection
#         residual_feature = context_features + non_local_feature

#     box_features = temporal_roi_cropping(residual_feature, shifted_rois, cur_b_idx, BOX_CROP_SIZE)

#     i3d_tail_feats = self.i3d_tail_model(box_features)

#     return i3d_tail_feats

# def non_local_ROI_model_v3(self, context_features, shifted_rois, cur_b_idx, BOX_CROP_SIZE):
#     '''
#     roi_box_features: bounding box features extracted on detected people
#     context_features: main feature map extracted from full frame
#     cur_b_idx: Batch - Roi mapping ex: 5 rois, 3 batch segments then an example [0,0,1,1,2]
#     '''
#     with tf.variable_scope('Non_Local_Block'):
#         # _, Tr, Hr, Wr, Cr = roi_box_features.shape.as_list()
#         # R = tf.shape(roi_box_features)[0]
#         _, Tc, Hc, Wc, Cc = context_features.shape.as_list()
#         B = tf.shape(context_features)[0]

#         feature_map_channel = Cc / 2

#         roi_embedding = tf.layers.conv3d(context_features, filters=feature_map_channel, kernel_size=[1,1,1], padding='SAME', activation=tf.nn.relu, name='RoiEmbedding')
#         context_embedding = tf.layers.conv3d(context_features, filters=feature_map_channel, kernel_size=[1,1,1], padding='SAME', activation=tf.nn.relu, name='ContextEmbedding')

#         context_response = tf.layers.conv3d(context_features, filters=feature_map_channel, kernel_size=[1,1,1], padding='SAME', activation=tf.nn.relu, name='ContextRepresentation')


#         # reshape so that we can use matrix multiplication to calculate attention mapping
#         roi_emb_reshaped = tf.reshape(roi_embedding, shape=[B, Tc*Hc*Wc, feature_map_channel])
#         context_emb_reshaped = tf.reshape(context_embedding, shape=[B, Tc*Hc*Wc, feature_map_channel])
#         context_res_reshaped = tf.reshape(context_response, shape=[B, Tc*Hc*Wc, feature_map_channel])

#         emb_mtx = tf.matmul(roi_emb_reshaped, context_emb_reshaped, transpose_b=True) # [B,Tc*Hc*Wc, Tc*Hc*Wc]

#         emb_mtx_spatial = tf.reshape(emb_mtx, [B, Tc, Hc, Wc, Tc*Hc*Wc])
#         roi_emb_mtx = temporal_roi_cropping(emb_mtx_spatial, shifted_rois, cur_b_idx, BOX_CROP_SIZE)
#         R = tf.shape(roi_emb_mtx)[0]
#         roi_emb_mtx_reshaped = tf.reshape(roi_emb_mtx, [R, Tc*BOX_CROP_SIZE[0]*BOX_CROP_SIZE[1], Tc*Hc*Wc])

#         embedding_attention = tf.nn.softmax(roi_emb_mtx_reshaped, name='EmbeddingNormalization')

#         # replicate context res to that batch dimension matches the no of rois
#         context_res_gathered = tf.gather(context_res_reshaped, cur_b_idx, axis=0, name='ContextResGather')

#         attention_response = tf.matmul(embedding_attention, context_res_gathered) # [R, Tr*Hr*Wr, feature_map_channel]

#         attention_response_org_shape = tf.reshape(attention_response, [R, Tc, BOX_CROP_SIZE[0], BOX_CROP_SIZE[1], feature_map_channel]) # Tr = Tc

#         # blow it up to original feature dimension
#         non_local_feature = tf.layers.conv3d(attention_response_org_shape, filters=Cc, kernel_size=[1,1,1], padding='SAME', activation=tf.nn.relu, name='NonLocalFeature')

#         # Residual connection
#         roi_box_features = temporal_roi_cropping(context_features, shifted_rois, cur_b_idx, BOX_CROP_SIZE)
#         residual_feature = roi_box_features + non_local_feature

#     # box_features = temporal_roi_cropping(residual_feature, shifted_rois, cur_b_idx, BOX_CROP_SIZE)

#     i3d_tail_feats = self.i3d_tail_model(residual_feature)

#     return i3d_tail_feats


def non_local_block(self, context_features):
    with tf.variable_scope('Non_Local_Block'):
        # _, Tr, Hr, Wr, Cr = roi_box_features.shape.as_list()
        # R = tf.shape(roi_box_features)[0]
        _, Tc, Hc, Wc, Cc = context_features.shape.as_list()
        B = tf.shape(context_features)[0]

        feature_map_channel = Cc / 2

        roi_embedding = tf.layers.conv3d(context_features, filters=feature_map_channel, kernel_size=[1,1,1], padding='SAME', activation=tf.nn.relu, name='RoiEmbedding')
        context_embedding = tf.layers.conv3d(context_features, filters=feature_map_channel, kernel_size=[1,1,1], padding='SAME', activation=tf.nn.relu, name='ContextEmbedding')

        context_response = tf.layers.conv3d(context_features, filters=feature_map_channel, kernel_size=[1,1,1], padding='SAME', activation=tf.nn.relu, name='ContextRepresentation')


        # reshape so that we can use matrix multiplication to calculate attention mapping
        roi_emb_reshaped = tf.reshape(roi_embedding, shape=[B, Tc*Hc*Wc, feature_map_channel])
        context_emb_reshaped = tf.reshape(context_embedding, shape=[B, Tc*Hc*Wc, feature_map_channel])
        context_res_reshaped = tf.reshape(context_response, shape=[B, Tc*Hc*Wc, feature_map_channel])

        emb_mtx = tf.matmul(roi_emb_reshaped, context_emb_reshaped, transpose_b=True) # [R,Tr*Hr*Wr, Tc*Hc*Wc]

        embedding_attention = tf.nn.softmax(emb_mtx, name='EmbeddingNormalization')

        attention_response = tf.matmul(embedding_attention, context_res_reshaped) # [R, Tr*Hr*Wr, feature_map_channel]

        attention_response_org_shape = tf.reshape(attention_response, [B, Tc, Hc, Wc, feature_map_channel])

        # blow it up to original feature dimension
        non_local_feature = tf.layers.conv3d(attention_response_org_shape, filters=Cc, kernel_size=[1,1,1], padding='SAME', activation=tf.nn.relu, name='NonLocalFeature')

    return non_local_feature

def multi_non_local_block_roi(self, context_features, shifted_rois, cur_b_idx, BOX_CROP_SIZE):
    no_non_locals = 5
    non_local_feats = []
    for ii in range(no_non_locals):
        with tf.variable_scope('NL_%i' % ii):
            nl_feat = self.non_local_block(context_features)
            non_local_feats.append(nl_feat)
    
    residual_connections = tf.add_n(non_local_feats + [context_features], name='Residual')

    box_features = temporal_roi_cropping(residual_connections, shifted_rois, cur_b_idx, BOX_CROP_SIZE)

    i3d_tail_feats = self.i3d_tail_model(box_features)

    return i3d_tail_feats


def multiscale_basic_model(self, end_points, shifted_rois, cur_b_idx, cropsize):
    feature_map_ids = ['Mixed_3c', 'Mixed_4e', 'Mixed_5c']
    
    features = [end_points[mapid] for mapid in feature_map_ids]
    cropped_features = [temporal_roi_cropping(feature_map, shifted_rois, cur_b_idx, cropsize) for feature_map in features]

    flattened_features = [basic_model(cropfeat) for cropfeat in cropped_features]
    # import pdb;pdb.set_trace()
    class_feats = tf.concat(flattened_features, -1)

    return class_feats

def roi_object_relation_model(self, roi_box_features, context_features, cur_b_idx):
    '''
    roi_box_features: bounding box features extracted on detected people
    context_features: main feature map extracted from full frame
    cur_b_idx: Batch - Roi mapping ex: 5 rois, 3 batch segments then an example [0,0,1,1,2]
    '''
    with tf.variable_scope('Non_Local_Block'):
        _, Tr, Hr, Wr, Cr = roi_box_features.shape.as_list()
        R = tf.shape(roi_box_features)[0]
        _, Tc, Hc, Wc, Cc = context_features.shape.as_list()
        B = tf.shape(context_features)[0]


        feature_map_channel = Cr / 2

        roi_embedding = tf.layers.conv3d(roi_box_features, filters=feature_map_channel, kernel_size=[Tr,Hr,Wr], padding='SAME', activation=tf.nn.relu, name='RoiEmbedding')
        
        context_embedding = tf.layers.conv3d(context_features, filters=feature_map_channel, kernel_size=[5,5,5], padding='SAME', activation=tf.nn.relu, name='ContextEmbedding')

        context_response = tf.layers.conv3d(context_features, filters=feature_map_channel, kernel_size=[1,1,1], padding='SAME', activation=tf.nn.relu, name='ContextRepresentation')

        # Number of rois(R) is larger than number of batches B as from each segment we extract multiple rois
        # we need to gather batches such that rois are assigned to correct context features that they were extracted from
        context_embedding_gathered = tf.gather(context_embedding, cur_b_idx, axis=0, name='ContextEmbGather')
        context_response_gathered = tf.gather(context_response, cur_b_idx, axis=0, name='ContextResGather')
        # now they have R as first dimensions

        # reshape so that we can use matrix multiplication to calculate attention mapping
        roi_emb_reshaped = tf.reshape(roi_embedding, shape=[R, Tr*Hr*Wr, feature_map_channel])
        context_emb_reshaped = tf.reshape(context_embedding_gathered, shape=[R, Tc*Hc*Wc, feature_map_channel])
        context_res_reshaped = tf.reshape(context_response_gathered, shape=[R, Tc*Hc*Wc, feature_map_channel])

        emb_mtx = tf.matmul(roi_emb_reshaped, context_emb_reshaped, transpose_b=True) # [R,Tr*Hr*Wr, Tc*Hc*Wc]

        embedding_attention = tf.nn.softmax(emb_mtx, name='EmbeddingNormalization')

        attention_response = tf.matmul(embedding_attention, context_res_reshaped) # [R, Tr*Hr*Wr, feature_map_channel]

        attention_response_org_shape = tf.reshape(attention_response, [R, Tr, Hr, Wr, feature_map_channel])

        # blow it up to original feature dimension
        non_local_feature = tf.layers.conv3d(attention_response_org_shape, filters=Cr, kernel_size=[1,1,1], padding='SAME', activation=tf.nn.relu, name='NonLocalFeature')

        # Residual connection
        residual_feature = roi_box_features + non_local_feature

    i3d_tail_feats = self.i3d_tail_model(residual_feature)

    return i3d_tail_feats



####### COMMON LAYERS ####### 

def combine_batch_rois(rois, labels, ):#no_dets):
    '''
    rois is [BATCH, MAX_ROIS, 4]
    labels is [BATCH, MAX_ROIS, NUM_CLASSES]
    no_dets is [BATCH] indicating the number of nonzero terms in each batch dim
    '''
    B = tf.shape(rois)[0]
    num_classes = labels.shape[2]
    max_rois = labels.shape[1]
     
    rois_all = tf.reshape(rois, [-1, 4])
    labels_all = tf.reshape(labels, [-1,num_classes])
 
    # We need a way to map each rois to its sample
    # samples have their batch indices
    # since we have constant no of max_rois I can just generate the indices
    batch_range = tf.expand_dims(tf.range(B),axis=1) # size Bx1
    batch_idx_tiled = tf.tile(batch_range, [1, max_rois]) # size BxMAX_ROIS
    batch_indices_all = tf.reshape(batch_idx_tiled, [-1])
 
    # TODO
    # Now everything has the first dimension with size B*MAX_ROIS
    # we have to find the non-zero roi terms, I can filter these out using no dets
 
    # starting_indices = tf.range(B) * MAX_ROIS
    # end_indices = starting_indices + no_dets
 
    # Check if all roi boxes are zero
    zero_indices = tf.equal(rois_all, 0.0) # same shape as rois_all
    zero_rows = tf.logical_and(zero_indices[:,0], zero_indices[:,1])
    zero_rows = tf.logical_and(zero_rows, zero_indices[:,2])
    zero_rows = tf.logical_and(zero_rows, zero_indices[:,3])
 
    non_zero_rows = tf.logical_not(zero_rows)
     
    # remove the zero rows
    # now everything will have num_boxes first dimension
    rois_nonzero = tf.boolean_mask(rois_all, non_zero_rows, axis=0)
    labels_nonzero = tf.boolean_mask(labels_all, non_zero_rows, axis=0)
    batch_indices_nonzero = tf.boolean_mask(batch_indices_all, non_zero_rows, axis=0)
 
    return rois_nonzero, labels_nonzero, batch_indices_nonzero

def temporal_roi_cropping(features, rois, batch_indices, crop_size):
    ''' features is of shape [Batch, T, H, W, C]
        rois [num_boxes, TEMP_RESOLUTION, 4] or [num_boxes, 4] depending on temp_rois flag
        batch_indices [num_boxes]
    '''
    # import pdb;pdb.set_trace()
    B = tf.shape(features)[0]
    _, T, H, W, C = features.shape.as_list()
    num_boxes = tf.shape(rois)[0]
 
    # if temp_rois:
    #     # slope = (T-1) / tf.cast(TEMP_RESOLUTION - 1, tf.float32)
    #     # indices = tf.cast(slope * tf.range(TEMP_RESOLUTION, dtype=tf.float32), tf.int32)
    #     slope = (TEMP_RESOLUTION-1) / float(T - 1)
    #     indices = (slope * np.arange(T)).astype(np.int32)
    #     temporal_rois = tf.gather(rois,indices,axis=1,name='temporalsampling')
    # else:
    #     # use the keyframe roi for all time indices
    #     temporal_rois = tf.expand_dims(rois, axis=1)
    #     temporal_rois = tf.tile(temporal_rois, [1, T, 1])
    temporal_rois = tf.expand_dims(rois, axis=1)
    temporal_rois = tf.tile(temporal_rois, [1, T, 1])
 
    # since we are flattening the temporal dimension and batch dimension
    # into a single dimension, we need new batch_index mapping
     
    # batch_indices = [0,0,1,1,2]
    temporal_mapping = batch_indices * T # gives the starting point for each sample in batch
    # temporal_mapping = [0,0,16,16,32]
     
    temporal_mapping = tf.expand_dims(temporal_mapping, axis=1)
    # temporal_mapping = [0,0,16,16,32]
     
    temporal_mapping = tf.tile(temporal_mapping, [1, T])
    #   [[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    #    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    #    [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
    #    [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
    #    [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
    #    [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]],
     
    temporal_mapping = temporal_mapping + tf.range(T)
    #    [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
    #    [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
    #    [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
    #    [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
    #    [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
    #    [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]],
 
    temporal_mapping = tf.reshape(temporal_mapping, [-1])
    #   [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0,
    #     1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
    #    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 16, 17, 18,
    #    19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 16, 17, 18, 19,
    #    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
    #    37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
 
 
    # combine temporal dimension with batch dimension
    #stacked_features = tf.transpose(features, perm=[1, 0, 2, 3, 4])
    stacked_features = tf.reshape(features, [-1, H, W, C])
 
    # combine rois and mappings
    #stacked_rois = tf.transpose(rois, perm=[1,0,2])
    stacked_rois = tf.reshape(temporal_rois, [-1, 4])
 
    #stacked_mapping = tf.transpose(mapping, perm=[1,0])
    stacked_mapping = tf.reshape(temporal_mapping, [-1])
 
    ## cropped boxes 
    cropped_boxes = tf.image.crop_and_resize(image=stacked_features, 
                                             boxes=stacked_rois,
                                             box_ind=stacked_mapping,
                                             crop_size=crop_size
                                             )

    # ## Bilinearly crop first and then take max pool. This in theory would work better with sparse feats
    # double_size = [cc*2 for cc in crop_size]
    # bilinear_cropped_boxes = tf.image.crop_and_resize(  image=stacked_features, 
    #                                                     boxes=stacked_rois,
    #                                                     box_ind=stacked_mapping,
    #                                                     crop_size=double_size
    #                                                     )
    # cropped_boxes = tf.layers.max_pooling2d(bilinear_cropped_boxes, [2,2], [2,2], padding='valid')

    # now it has shape B*T, crop size
    # cropped_boxes = tf.Print(cropped_boxes, [tf.shape(cropped_boxes)], 'cropped shape')
    # unrolled_boxes = tf.reshape(cropped_boxes, [T, num_boxes, crop_size[0], crop_size[1], C])
    unrolled_boxes = tf.reshape(cropped_boxes, [num_boxes, T, crop_size[0], crop_size[1], C])
 
    # swap the boxes and time dimension
    # boxes = tf.transpose(unrolled_boxes, perm=[1, 0, 2, 3, 4])
    boxes = unrolled_boxes
 
    return boxes #, stacked_features


def generate_temporal_rois(rois, mapping, T):
    ''' 
    rois [num_boxes, 4]
    mapping [num_boxes]
    T is the no frames in temporal dimension
    '''
    num_boxes = rois.shape[0]
    new_rois = np.zeros([num_boxes, T, 4])
    new_mapping = np.zeros([num_boxes, T])
    for t in range(T):
        new_rois[:,t,:] = rois

    for b in range(num_boxes):
        new_mapping[b, :] = np.arange(T) + mapping[b] * T

    return new_rois, new_mapping
