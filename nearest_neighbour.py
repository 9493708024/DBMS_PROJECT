import numpy as np
import scipy.io
from sklearn.decomposition import PCA
from scipy.stats import ortho_group
from scipy import linalg
import math
import scipy
import matplotlib.pyplot as plt
import cPickle

def unpickle(file):  #for converting data to to images
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    data = dict['data']
    imgs = np.transpose(np.reshape(data,(-1,32,32,3), order='F'),axes=(0,2,1,3)) #order batch,x,y,color
    y = np.asarray(dict['labels'], dtype='uint8')
    return y, imgs


def set_bit(v, index, x):
  mask = 1 << index   # Compute mask, an integer with just bit 'index' set.
  if x:
    v |= mask         # If x was True, set the bit indicated by the mask.
  return v            # Return the result, we're done.


def ITQ(Y,num_iter):
	num_bits = Y.shape[1]
	R = ortho_group.rvs(dim=num_bits)

	for x in range(num_iter):
		#print Y.shape,R.shape
		Z = np.dot(Y,R)
		UX = np.full((Z.shape[0],Z.shape[1]),-1)
		UX[Z>=0] = 1
		C = np.dot(UX.T,Y)
		UB,sigma,UA = linalg.svd(C)
		R = np.dot(UA,UB.T)
	B = UX
	B[B<0] = 0
	return B,R

def compactbits(b):
	num_samples = b.shape[0]
	num_bits    = b.shape[1]
	num_words   = math.ceil(num_bits/8)
	temp        = np.zeros((num_samples,int(num_words)),dtype=np.uint8)
	for i in range(num_bits):
		k = int(math.floor(i/8))
		for x in range(num_samples):
			temp[x,k] = set_bit(temp[x,k],int(i%8),b[x,i])
	return temp		 

def hammingDist(B1,B2):
    y = np.array([0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,1,2,2,3,2,3,\
        3,4,2,3,3,4,3,4,4,5,1,2,2,3,2,3,3,4,2,3,3,4,\
        3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,1,2,\
        2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,\
        3,4,4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,4,5,\
        5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,1,2,2,3,\
        2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,\
        4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,\
        3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,2,3,3,4,3,4,\
        4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,\
        5,6,6,7,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,\
        5,6,5,6,6,7,5,6,6,7,6,7,7,8], dtype=np.uint16)

    n1 = B1.shape[0]
    print B1.shape
    n2,num_words = B2.shape
    print n2
    print B2.shape
    temp = np.zeros((n1,n2),'uint16')
    print temp.shape
    for i in range(n1):
        for j in range(n2):
            for k in range(num_words):
                temp[i,j] = temp[i,j]+y[B1[i,k]^B2[j,k]]

    
    return temp            

data = scipy.io.loadmat('cifar_10yunchao.mat')

data = data.get('cifar10')
data = np.array(data)
print data.shape

num_bits = 32;  

num_train  = 59000 
train_data = data[0:num_train,:-1]
test_data  = data[num_train:,:-1]

print train_data.shape

XX = data[:,:-1]
m = XX.mean(axis=0)
XX = XX - XX.mean(axis=0)

print XX.shape
pca = PCA(n_components=num_bits, svd_solver='full')
pca.fit(XX[:num_train])
XX = pca.transform(XX)

final,R = ITQ(XX[:59000],50)
XX = np.dot(XX,R)
final = np.zeros(XX.shape)
final[XX>=0] = 1;

print final.shape,final
final = compactbits(final)
B1 = final[:num_train]
B2 = final[num_train:]

Dist = hammingDist(B2,B1)
print Dist.shape

Dist = np.argsort(Dist)

y1, imgs1 = unpickle('cifar-10-batches-py/data_batch_1')
y2, imgs2 = unpickle('cifar-10-batches-py/data_batch_2')
y3, imgs3 = unpickle('cifar-10-batches-py/data_batch_3')
y4, imgs4 = unpickle('cifar-10-batches-py/data_batch_4')
y5, imgs5 = unpickle('cifar-10-batches-py/data_batch_5')
y6, imgs6 = unpickle('cifar-10-batches-py/test_batch')

final_data = np.concatenate((imgs1,imgs2,imgs3,imgs4,imgs5,imgs6), axis=0)
can = np.zeros((320,416,3),dtype='uint8') 

index = [59017]
index.extend(Dist[17])
print len(index)
for i in range(10):
    for j in range(13):
        can[32*i:32*(i+1),32*j:32*(j+1),:] = final_data[index[10*i+j]]

fig = plt.figure(figsize=(7,14))
fig.xticks=[]
plt.imshow(can)
plt.show()
print "hello"
scipy.misc.imsave('cifar_examples.jpg', can)



