import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def FetchData(hours):
	data = pd.read_csv("train.csv").as_matrix()
	X = np.empty([0,9])
	Y = np.empty([0,1])
	
	#PM2.5 is at 10th of Row
	for i in range(0, data.shape[0], 18):
		
		pm = data[ i+9, 3: ].astype(float)
		for j in range(0, 24, hours+1):
			if j+hours > 24:
				break
			xi = pm[j:j+hours].reshape(1,9)
			yi = np.array([pm[j+hours]]).reshape(1,1)
			X = np.concatenate( (X,xi), axis=0)
			Y = np.concatenate( (Y,yi), axis=0)
	
	X = np.concatenate( (X, np.ones([X.shape[0],1])), axis=1) #add one 
	return X,Y

def GD(X, Y, lr, iter): #1e-7
	ls_rec  = []
	iter_rec = []
	theta = np.zeros([X.shape[1],1])
	for i in range(iter):
		gd_theta = -2* np.dot( X.transpose(), (Y-np.dot(X,theta)) ) #-2*X^T(Y-X*theta)
		theta = theta - lr * gd_theta
		
		Loss = np.sqrt(np.sum((Y-np.dot(X,theta))**2)/X.shape[0])
		ls_rec.append(Loss)
		iter_rec.append(i)
	return theta ,ls_rec,iter_rec


def SGD_ADAGRAD(X, Y, lr, iter): #1e-4
	ls_rec  = []
	iter_rec = []
	theta = np.zeros([X.shape[1],1])
	
	for i in range(iter):
		n_theta = 0
		for j in range(X.shape[0]):
			gd_theta = -2* X[j].transpose() * (Y[j]-np.dot(X[j],theta))
			gd_theta = gd_theta.reshape(gd_theta.shape[0],1)
			n_theta = n_theta + gd_theta**2
			theta = theta - lr/np.sqrt(n_theta) * gd_theta
			
		Loss = np.sqrt(np.sum((Y-np.dot(X,theta))**2)/X.shape[0])
		ls_rec.append(Loss)
		iter_rec.append(i)
	return theta,ls_rec,iter_rec

def SGD(X, Y, lr, iter): #1e-7
	ls_rec  = []
	iter_rec = []
	theta = np.zeros([X.shape[1],1])
	
	for i in range(iter):
		n_theta = 0
		for j in range(X.shape[0]):
			gd_theta = -2* X[j].transpose() * (Y[j]-np.dot(X[j],theta))
			gd_theta = gd_theta.reshape(gd_theta.shape[0],1)
			theta = theta - lr* gd_theta
			
		Loss = np.sqrt(np.sum((Y-np.dot(X,theta))**2)/X.shape[0])
		ls_rec.append(Loss)
		iter_rec.append(i)
	return theta,ls_rec,iter_rec



def main():
	X,Y = FetchData(9)
	theta, ls_rec, iter_rec = SGD(X, Y, 1e-9, 10000)
	
	result = np.dot(X,theta)
	cmp = np.concatenate((Y,result),axis=1)
	print(cmp)
	print(ls_rec[-1])	
	
	plt.plot(iter_rec, ls_rec)
	plt.xlabel("iter")
	plt.ylabel("Loss")
	plt.savefig("SGD.png")
	
	theta, ls_rec, iter_rec = SGD_ADAGRAD(X, Y, 1e-5, 10000)
	
	result = np.dot(X,theta)
	cmp = np.concatenate((Y,result),axis=1)
	print(cmp)
	print(ls_rec[-1])	
	
	plt.plot(iter_rec, ls_rec)
	plt.xlabel("iter")
	plt.ylabel("Loss")
	plt.savefig("SGD_ADAGRAD.png")
	
	
	
	
if __name__ == "__main__":
	main()