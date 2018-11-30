from src.mnist_loader import load_data_wrapper
from src.network import Network

if __name__=='__main__':
   train,vaild,test = load_data_wrapper()
   net = Network([784,30,10])
   net.SGD(train,30 ,10 ,3.0,test_data=test)

