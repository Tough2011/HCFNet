#coding:utf-8
import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data
class FC:
    def __init__(self, n_hidden_0, n_hidden_1,  batchSize, CName,WName):
       # self.s = s

        self.batchSize = batchSize
        self.n_hidden_0 = n_hidden_0  # 输出值
        self.n_hidden_1 = n_hidden_1  #  输入值
        tsvalue= 2
        patch = int((n_hidden_0-tsvalue)/2)

        defaut_initializer0 = tf.random_uniform_initializer(minval=0.2, maxval=2)
        defaut_initializer1 = tf.random_uniform_initializer(minval=3, maxval=10)
        defaut_initializer2 = tf.random_uniform_initializer(minval=2, maxval=2.5)


        p0 = tf.get_variable('p0', shape=[1, tsvalue], initializer=defaut_initializer0, dtype=tf.float32)
        p1 = tf.get_variable('p1', shape=[1, patch], initializer=defaut_initializer1,   dtype=tf.float32)
        p2 = tf.get_variable('p2', shape=[1, patch], initializer=defaut_initializer2, dtype=tf.float32)
        self.p =tf.concat([p1,p0, p2], 1)

       # pp = tf.truncated_normal([1, self.n_hidden_0], mean=0.0, stddev=1.0, dtype=tf.float32, seed=10)
       # self.p = tf.Variable(pp, name='p0')

        #self.p =   tf.Variable(tf.ones([1, n_hidden_0]), name="v2")
        #self.p = 3

        Bias = tf.truncated_normal([1,self.n_hidden_0], mean=0.0, stddev=1.0, dtype=tf.float32, seed=100)
        self.b = tf.Variable(Bias, name='layer1_B')

        defaut_initializerC = tf.contrib.layers.xavier_initializer()
        self.C =tf.get_variable(shape=[self.n_hidden_0, self.n_hidden_1 ],initializer=defaut_initializerC ,name=CName)

        defaut_initializerW = tf.contrib.layers.xavier_initializer()


        self.W = tf.get_variable(shape=[self.n_hidden_0, self.n_hidden_1], initializer=defaut_initializerW, name=WName)

        s1 = tf.constant(0, dtype=tf.float32, shape=[1, self.n_hidden_0/4])
        s2 = tf.constant(1, dtype=tf.float32, shape=[1, self.n_hidden_0 / 4])
        s3 = tf.constant(0, dtype=tf.float32, shape=[1, self.n_hidden_0 / 4])
        s4 = tf.constant(0, dtype=tf.float32, shape=[1, self.n_hidden_0 / 4])


        self.s = tf.concat([s1,s2,s3,s4],1)

    def forward(self, in_data):
        # in_data1 = tf.expand_dims(in_data, axis=1)
        # temp = tf.reduce_sum(tf.transpose((in_data1 - self.C) * self.W,perm=[0,2, 1] ), 1)
        # Y1 = tf.pow(tf.abs(temp), self.p)
        # Y2 = tf.pow(tf.sign(temp), self.s)
        # self.topVal = tf.subtract(tf.multiply(Y1, Y2), self.b)
        #
        #consistent with the equal
        in_data1 = tf.expand_dims(in_data, axis=1)
        q = tf.transpose(tf.multiply(tf.subtract(in_data1, self.C), self.W), perm=[0, 2, 1])
        q1 = tf.reduce_sum(tf.multiply(tf.pow(tf.sign(q), self.s), tf.pow(tf.abs(q), self.p)), axis=1)
        self.topVal = tf.subtract(q1, self.b)

        return self.topVal
def weight_variable(shape,name):
    defaut_initializer = tf.contrib.layers.xavier_initializer()
    initial=  tf.get_variable(name=name, shape=shape, initializer=defaut_initializer)
    return initial
def bias_variable(shape,name ):
    defaut_initializer = tf.contrib.layers.xavier_initializer()
    initial =  tf.get_variable(name=name, shape=shape, initializer=defaut_initializer)
    return  initial
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1],
                          strides = [1, 2, 2, 1], padding = 'SAME')

start = time.clock()

mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)
x = tf.placeholder(tf.float32,[None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])
W_conv1 = weight_variable([5, 5, 1, 32],name='W_conv1')
b_conv1 = bias_variable([32], 'b_conv1')

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 16],name='W_conv2')
b_conv2 = bias_variable([16], 'b_conv2')

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 16])
batch_size=2048
n_hidden_0 = 7 * 7 * 16  # 输入
#n_hidden_1 = 2048  # 输出

n_hidden_1 = 128  # 输出
p = 3

#__init__(self, n_hidden_0, n_hidden_1,  batchSize, p,s):
#self.n_hidden_0 = n_hidden_0  # 输出值
#self.n_hidden_1 = n_hidden_1  # 输入值
fc = FC(n_hidden_1,  n_hidden_0, batch_size,  "CNameFc1","WNameFc1")
z1 = tf.nn.leaky_relu( fc.forward(h_pool2_flat) )  #  只需要一层 ，是对样本的特征空间的流行结构 进行刻画、表示，这是不同于传统神经元的地方，传统神经元更重要的是起到映射作用，而我们的不是，
# 我们用拓扑的测度进行刻画， 是形象几何的精髓思想， 拓扑不变性、稳定性



classNumb = n_hidden_1
n_hidden_3=10
W_fc3 = weight_variable([n_hidden_1 , 10    ],name='W_fc3')
b_fc3= bias_variable([10],'b_fc3')
#y_conv = tf.nn.softmax( tf.add(tf.matmul(z1,W_fc3),b_fc3  ) )   # 对输出的距离 进行评估，起到和最小二乘法同样的功能， 得到的是个分布，不是具体的距离

y_conv = tf.add(tf.matmul(z1,W_fc3),b_fc3  )
y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))

L = y_ * tf.square(tf.maximum(0., 0.9 - y_conv)) + 0.5 * (1 - y_) * tf.square(tf.maximum(0., y_conv - 0.1))
margLoss =  tf.reduce_mean(tf.reduce_sum(L, 1))


train_vars=tf.trainable_variables()
var_list1=[var for var in train_vars  if  var.name=='p0:0' or   var.name=='p1:0' or    var.name=='p2:0']
var_list2=[var for var in train_vars  if  var.name!='p0:0' or   var.name!='p1:0' or    var.name!='p2:0'  ]

#train_step = tf.train.AdamOptimizer(1e-4).minimize(margLoss)


opt1 = tf.train.AdamOptimizer(0.01)
opt2 = tf.train.AdamOptimizer(1e-4)
grads = tf.gradients(margLoss, var_list1 + var_list2)
grads1 = grads[:len(var_list1)]
grads2 = grads[len(var_list1):]
train_op1 = opt1.apply_gradients(zip(grads1, var_list1))
train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
train_step = tf.group(train_op1, train_op2)



correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

with tf.Session() as  sess :
    sess.run(tf.initialize_all_variables())
    for i in range(40000):
        batch = mnist.train.next_batch(32)
        train_step.run(session = sess, feed_dict = {x:batch[0], y_:batch[1]})
        if i %20 == 0:

            train_accuracy = accuracy.eval(session = sess,
                                           feed_dict = {x:batch[0], y_:batch[1]})
            batch_val = mnist.test.next_batch(32)
            val_accuracy = accuracy.eval(session = sess,
                                           feed_dict = {x:batch_val[0], y_:batch_val[1]})
            print('====================================================>>>>>')
            print('=---')
            sess.run(fc.p)
            print("step %d, train_accuracy %g" %(i, train_accuracy))
            print("step %d, val_accuracy %g" % (i, val_accuracy))
            print('====================================================>>>>>')

