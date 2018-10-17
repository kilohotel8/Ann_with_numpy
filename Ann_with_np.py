import numpy as np
import os
import urllib.request
import gzip


class Ann_with_np:
    def __init__(self):
        dir = os.getcwd()
        urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/' + 'train-images-idx3-ubyte.gz',
                                   dir + '/train-images-idx3-ubyte.gz')
        urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/' + 'train-labels-idx1-ubyte.gz',
                                   dir + '/train-labels-idx1-ubyte.gz')
        urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/' + 't10k-images-idx3-ubyte.gz',
                                   dir + '/t10k-images-idx3-ubyte.gz')
        urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/' + 't10k-labels-idx1-ubyte.gz',
                                   dir + '/t10k-labels-idx1-ubyte.gz')
        self.__xdata = self.__extractdata('train-images-idx3-ubyte.gz',16,784,60000).reshape(60000,784)/256
        self.__ydata = self.__onehot(self.__extractdata('train-labels-idx1-ubyte.gz',8,1,60000))
        self.__xdata_test = self.__extractdata('t10k-images-idx3-ubyte.gz',16,784,10000).reshape(10000,784)/256
        self.__ydata_test =  self.__onehot(self.__extractdata('t10k-labels-idx1-ubyte.gz',8,1,10000))
        self.__xdata,self.__ydata = self.__batching(self.__xdata,self.__ydata)
        self.w1 = np.random.randn(784,256) / np.sqrt(392)
        self.w2 = np.random.randn(256,32) / np.sqrt(128)
        self.w3 = np.random.randn(32,10) / np.sqrt(16)
        self.b1 = np.random.randn(256) / np.sqrt(128)
        self.b2 = np.random.randn(32) / np.sqrt(16)
        self.b3 = np.random.randn(10) / np.sqrt(5)
        print("model is created")

    def __extractdata(self,filename, readdata, datasize, num):
        with gzip.open(filename) as bytestream:
            bytestream.read(readdata)
            buf = bytestream.read(datasize * num)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            return data

    def __onehot(self,ydata):
        output = np.zeros(shape=(len(ydata), 10))
        for i in range(len(ydata)):
            output[i][int(ydata[i])] = 1
        return output

    def __batching(self,xdata,ydata,number_of_input = 60000,batch=200):
        total_batch = np.int(number_of_input / batch)
        x_batch = []
        y_batch = []
        for i in range(total_batch):
            x_batch.append(xdata[i * batch:(i + 1) * batch])
            y_batch.append(ydata[i * batch:(i + 1) * batch])
        return x_batch,y_batch

    @property
    def xdata(self):
        return self.__xdata

    @property
    def ydata(self):
        return self.__ydata

    @property
    def xdata_test(self):
        return self.__xdata_test

    @property
    def ydata_test(self):
        return self.__ydata_test


    def learning(self,xdata,ydata,num_epoch = 15,learning_rate = 0.01):
        for epoch in range(num_epoch):
            for batch in np.arange(len(xdata)):
                ######## forward ###########################
                x_batch = xdata[batch]
                y_batch = ydata[batch]
                beforeactivation1 = np.dot(x_batch, self.w1) + self.b1
                hidden1 = np.maximum(beforeactivation1, 0.1 * beforeactivation1)
                beforeactivation2 = np.dot(hidden1, self.w2) + self.b2
                hidden2 = np.maximum(beforeactivation2, 0.1 * beforeactivation2)
                output = np.dot(hidden2, self.w3) + self.b3
                softmax = self.__applysoftmax(output)
                loss = np.sum(-y_batch*np.log(softmax),axis = 1)


                ######### backpropagation ##########
                error3 = output - y_batch
                d_loss_w3 = np.zeros([200,32,10])
                for i in range(200):
                    d_loss_w3[i] = hidden2[i].reshape(32,1) * error3[i].reshape(1,10)
                d_loss_b3 = error3
                d_activation2 = self.__d_LeakyRelu(np.dot(hidden1, self.w2))

                error2 = d_activation2 * np.matmul(error3,self.w3.T)
                d_loss_w2 = np.zeros([200, 256, 32])
                for i in range(200):
                    d_loss_w2[i] = hidden1[i].reshape(256,1) * error2[i].reshape(1,32)
                d_loss_b2 = error2
                d_activation1 = self.__d_LeakyRelu(np.dot(x_batch, self.w1))

                error1 = d_activation1 * np.matmul(error2,self.w2.T)
                d_loss_w1 = np.zeros([200, 784, 256])
                for i in range(200):
                    d_loss_w1[i] = x_batch[i].reshape(784,1) * error1[i].reshape(1,256)
                d_loss_b1 = error1

                #####updating################################################
                self.w3 = self.w3 - learning_rate*np.mean(d_loss_w3,axis=0)
                self.w2 = self.w2 - learning_rate*np.mean(d_loss_w2,axis=0)
                self.w1 = self.w1 - learning_rate*np.mean(d_loss_w1,axis=0)
                self.b3 = self.b3 - learning_rate*np.mean(d_loss_b3,axis=0)
                self.b2 = self.b2 - learning_rate*np.mean(d_loss_b2,axis=0)
                self.b1 = self.b1 - learning_rate*np.mean(d_loss_b1,axis=0)


                if batch %10 == 0:
                    print("epoch : {}, batch : {}, loss: {}".format(epoch,batch,np.mean(loss,axis = 0)))
        print("learning ends")

    def __applysoftmax(self,array):
        z = array - np.max(array)
        exp_output = np.exp(z)
        return exp_output / np.sum(exp_output,axis = -1).reshape(-1,1)

    def __d_LeakyRelu(self,array):
        array[array>0] = 1
        array[array<0] = 0.1
        return array

    def accuracy(self,xdata_test,ydata_test):
        correct = 0
        for i in np.arange(len(xdata_test)):
            hidden1 = np.maximum(np.dot(xdata_test[i], self.w1) + self.b1, 0.1 * np.dot(xdata_test[i], self.w1) + self.b1)
            hidden2 = np.maximum(np.dot(hidden1, self.w2) + self.b2, 0.1 * np.dot(hidden1, self.w2) + self.b2)
            output = np.matmul(hidden2, self.w3) + self.b3
            softmax = self.__applysoftmax(output)
            if (np.where(softmax[0] == softmax[0].max()) == np.where(ydata_test[i] == 1)):
               correct += 1

        print("accuracy: ", correct / len(ydata_test))


def main():
    nn = Ann_with_np()
    xdata, ydata, xdata_test, ydata_test = nn.xdata, nn.ydata, nn.xdata_test, nn.ydata_test
    nn.learning(xdata,ydata)
    nn.accuracy(xdata_test,ydata_test)

if __name__ == "__main__":
	main()