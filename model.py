import pickle
import tensorflow as tf

class Yolov2:
    def __init__(self , conv_weights_file , bn_weights_file , grid_size , num_anchors , num_classes , is_training):
        self.grid_height = grid_size
        self.grid_width = grid_size
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.is_training = is_training
        self.conv_weights = self.load_pickle(conv_weights_file)
        self.bn_weights = self.load_pickle(bn_weights_file)
        self.conv_index = 0
        self.bn_epsilon = 1e-05

    def load_pickle(self , file):
        with open(file , "rb") as content:
            return pickle.load(content)

    def batch_norm(self , inputs, n_out):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(inputs, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(self.is_training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
        return normed

    def init_weights(self, shape):
        weights = tf.truncated_normal(shape , stddev = 0.1)
        return tf.Variable(weights)

    def init_biases(self , shape):
        bias_shape = shape[3:]
        biases = tf.constant(0.1 , shape = bias_shape)
        return tf.Variable(biases)

    def conv_block(self, input , filter_shape , strides = 1 , padding = "SAME" , if_mp = False):
        conv_weights = self.init_weights(filter_shape)
        out_channels = filter_shape[-1]
        strides = [1,strides,strides,1]

        conv_layer = tf.nn.conv2d(input , conv_weights , strides = strides , padding = padding)
        bn_layer = self.batch_norm(conv_layer , out_channels)
        leaky_relu_layer = tf.nn.leaky_relu(bn_layer , alpha = 0.1)
        if if_mp:
            mp_layer =  tf.nn.max_pool(leaky_relu_layer , ksize = [1,2,2,1] , strides=  [1,2,2,1] , padding = "VALID")
            return mp_layer

        return leaky_relu_layer

    def conv_block_pretrained_weights(self , input , strides = 1 , padding = "SAME" , if_mp = False):
        layer_name = "conv_{}".format(self.conv_index)
        self.conv_index += 1
        strides = [1 , strides , strides , 1]

        conv_weights = self.conv_weights[layer_name]
        bn_weights = self.bn_weights[layer_name]
        mean = bn_weights["mean"]
        variance = bn_weights["variance"]
        beta = bn_weights["beta"]
        gamma = bn_weights["gamma"]

        conv_layer = tf.nn.conv2d(input , conv_weights , strides = strides , padding = padding)
        bn_layer = tf.nn.batch_normalization(conv_layer , mean , variance , beta , gamma , self.bn_epsilon)
        leaky_relu_layer = tf.nn.leaky_relu(bn_layer , alpha = 0.1)
        if if_mp:
            mp_layer = tf.nn.max_pool(leaky_relu_layer,  ksize = [1,2,2,1] , strides = [1,2,2,1] , padding = "VALID")
            return mp_layer

        return leaky_relu_layer

    def space_to_depth(self, input):
        return tf.space_to_depth(input , block_size = 2)


    def gen_model(self, input):
        layer_1  = self.conv_block_pretrained_weights(input , if_mp = True)
        layer_2  = self.conv_block_pretrained_weights(layer_1 , if_mp = True)
        layer_3  = self.conv_block_pretrained_weights(layer_2)
        layer_4  = self.conv_block_pretrained_weights(layer_3)
        layer_5  = self.conv_block_pretrained_weights(layer_4 , if_mp = True)
        layer_6  = self.conv_block_pretrained_weights(layer_5)
        layer_7  = self.conv_block_pretrained_weights(layer_6)
        layer_8  = self.conv_block_pretrained_weights(layer_7 , if_mp = True)
        layer_9  = self.conv_block_pretrained_weights(layer_8)
        layer_10 = self.conv_block_pretrained_weights(layer_9)
        layer_11 = self.conv_block_pretrained_weights(layer_10)
        layer_12 = self.conv_block_pretrained_weights(layer_11)
        layer_13 = self.conv_block_pretrained_weights(layer_12)
        route_1 = layer_13
        mp_layer_13 = tf.nn.max_pool(layer_13 , ksize = [1,2,2,1] , strides = [1,2,2,1] , padding = "VALID")
        layer_14 = self.conv_block_pretrained_weights(mp_layer_13)
        layer_15 = self.conv_block_pretrained_weights(layer_14)
        layer_16 = self.conv_block_pretrained_weights(layer_15)
        layer_17 = self.conv_block_pretrained_weights(layer_16)
        layer_18 = self.conv_block_pretrained_weights(layer_17)

        layer_19 = self.conv_block(layer_18 , [3,3,1024,1024])
        layer_20 = self.conv_block(layer_19 , [3,3,1024,1024])
        route_1_layer = self.conv_block(route_1 , [3,3,512,64])
        # space_to_depth will map the output channel from 64 to 256 (64*4)
        route_1_layer = self.space_to_depth(route_1_layer)
        layer_21 = tf.concat([route_1_layer , layer_20] , axis = -1)
        layer_22 = self.conv_block(layer_21 , [3,3,1280,1024])

        layer23_filtershape = [1,1,1024,(self.num_anchors * (4 + 1 + self.num_classes))]
        layer23_weights = self.init_weights(layer23_filtershape)
        layer23_biases = self.init_biases(layer23_filtershape)
        layer_23 = tf.nn.bias_add(tf.nn.conv2d(layer_22 , layer23_weights , strides = [1,1,1,1] , padding = "SAME") , layer23_biases)
        output = tf.reshape(layer_23 , [-1 , self.grid_height , self.grid_width , self.num_anchors, 4+1+self.num_classes])
        return output
