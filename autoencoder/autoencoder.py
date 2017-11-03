import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0/(fan_in+fan_out))
    high = constant * np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform((fan_in,fan_out),minval=low,maxval=high,dtype=tf.float32)

class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer = tf.train.AdamOptimizer(), scale = 0.1):
        self.n_input  = n_input  #输入变量数
        self.n_hidden = n_hidden #隐含层节点数
        self.transfer = transfer_function #隐含层激活函数
        self.scale = tf.placeholder(tf.float32) 
        self.training_scale = scale #高斯噪声系数
        network_weights = self._initialize_weights() #参数初始化
        self.weights = network_weights 
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        # 将输入x加上噪声，将加了噪声的输入与隐含层权重相乘，再加隐含层偏置，最后使用self.transfer对结果进行激活函数处理
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)),self.weights['w1']),self.weights['b1']))
        # 将隐含层的输出self.hidden乘上输出层的权重，再加输出层的偏置
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])
        # 自编码器的损失函数，平方误差
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x),2.0))
        # 训练操作为优化器self.optimizer对损失self.cost进行优化
        self.optimizer = optimizer.minimize(self.cost)
        init = tf.global_variables_initializer()
        # 创建Session 初始化自编码器的模型参数
        self.sess = tf.Session()
        self.sess.run(init)

    # 参数初始化函数
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype = tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype = tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype = tf.float32))
        return all_weights

    # 计算损失cost及执行一步训练的函数partial_fit
    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict = {self.x: X, self.scale : self.training_scale})
        return cost

    # 只求损失cost的函数
    def calc_total_cost(self,X):
        return self.sess.run(self.cost, feed_dict={self.x:X, self.scale:self.training_scale})

    # 返回自编码器隐含层的输出结果，提取一个接口来获取抽象后的特征，学习数据的高阶特征
    def transform(self, X):
        return self.sess.run(self.hidden,feed_dict = {self.x:X, self.scale: self.training_scale})

    # 将高阶特征复原为原始数据，和前面的transform函数将自编码器拆分为两个部分
    def generate(self, hidden = None):
        if hidden is None:
            hidden = np.random.normal(size = self.weights["b1"])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden:hidden})

    # 整体运行一遍复原过程，包括提取高阶特征和通过高阶特征恢复复原数据
    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict = {self.x:X, self.scale:self.training_scale})

    # 获取隐含层的权重w1
    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    # 获取隐含层的偏置系数b1
    def getBiases(self):
        return self.sess.run(self.weights['b1'])

mnist = input_data.read_data_sets('D:\\mldata\\mnist', one_hot = True)
def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test

# 获取随机block的函数
def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index : (start_index + batch_size)]

X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
n_samples = int(mnist.train.num_examples) # 总训练样本数
training_epochs = 20 # 最大的训练轮数
batch_size = 128 # 每次训练样本数
display_step = 1 #每隔一轮显示一次损失cost

# 创建一个AGN自编码器的实例
autoencoder = AdditiveGaussianNoiseAutoencoder(n_input = 784, n_hidden = 200, transfer_function = tf.nn.softplus, 
                                              optimizer = tf.train.AdamOptimizer(learning_rate = 0.001), scale = 0.01)

for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)
        cost = autoencoder.partial_fit(batch_xs)
        avg_cost = cost / n_samples * batch_size
    if epoch % display_step == 0:
        print ("Epoch:",'%04d' % (epoch + 1),"cost=","{:.9f}".format(avg_cost))