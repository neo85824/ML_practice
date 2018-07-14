import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def FetchData(hours, f_idx):
	data = pd.read_csv("train.csv").as_matrix()
	X = np.empty([0,2*hours])
	Y = np.empty([0,1])
	
	for i in range(0, data.shape[0], 18):
		#PM2.5 is at index  9  of data
		#PM10 is at 8
		pm_idx = 9
		other = data[ i+f_idx , 3:-1 ].astype(float) #other feature
		pm = data[ i+pm_idx, 3:-1 ].astype(float)  #pm2.5
		feat = np.concatenate( (other.reshape(1,other.size), pm.reshape(1,pm.size)), axis=0)  #concat
		for j in range(0, 24, hours+1):
			if j+hours > 24:
				break
			xi = feat[ :, j:j+hours]
			xi = xi.reshape(1,xi.size)
			yi = np.array([feat[-1, j+hours]]).reshape(1,1)
			X = np.concatenate( (X,xi), axis=0)
			Y = np.concatenate( (Y,yi), axis=0)
	
	X = np.concatenate( (X, np.ones([X.shape[0],1])), axis=1) #add one 
	return X,Y



def TestData(hours,f_idx):
	data = pd.read_csv("test.csv").as_matrix()
	X = np.empty([0,hours*2])
	Y = np.empty([0,1])
	for i in range(0, data.shape[0], 18):
		#PM2.5 is at index 9  of data
		#PM10 is at 8
		pm_idx = 9
		
		other = data[i+f_idx ,2:-1].reshape(1,hours).astype(float)
		pm = data[i+pm_idx, 2:-1].reshape(1,hours).astype(float)
		xi = np.concatenate( (other,pm), axis = 0)
		xi = xi.reshape(1,xi.size)
		yi = np.array([data[ i+pm_idx, -1]]).reshape(1,1).astype(float)
		X = np.concatenate( (X,xi), axis = 0)
		Y = np.concatenate( (Y,yi), axis = 0)

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


def SGD_ADAGRAD(X, Y, lr, iter): #1e-3
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

def SGD(X, Y, lr, iter): #1e-6
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
	X,Y = FetchData(8,8) #take PM10
	iter = 10000
	X_test, Y_test = TestData(8,8)
	
	#SGD ADAGRAD
	theta, ls_rec, iter_rec = SGD_ADAGRAD(X, Y, 1e-4, iter)
	
	result = np.dot(X_test,theta)
	Loss = np.sqrt(np.sum((Y_test-np.dot(X_test,theta))**2)/X_test.shape[0])
	print("train")
	print(ls_rec[-1])
	print("test")
	print(Loss)			
	
	plt.plot(iter_rec, ls_rec)
	plt.xlabel("iter")
	plt.ylabel("Loss")
	plt.savefig("SGD_ADAGRAD.png")
	
	"""
	
	#GD
	theta, ls_rec, iter_rec = GD(X, Y, 1e-6, iter)
	
	result = np.dot(X_test,theta)
	Loss = np.sqrt(np.sum((Y_test-np.dot(X_test,theta))**2)/X_test.shape[0])
	print("train")
	print(ls_rec[-1])
	print("test")
	print(Loss)	
	
	plt.plot(iter_rec, ls_rec)
	plt.xlabel("iter")
	plt.ylabel("Loss")
	plt.savefig("GD.png")
	
	
	#SGD
	theta, ls_rec, iter_rec = SGD(X, Y, 1e-5, iter)
	
	result = np.dot(X_test,theta)
	Loss = np.sqrt(np.sum((Y_test-np.dot(X_test,theta))**2)/X_test.shape[0])
	print("train")
	print(ls_rec[-1])
	print("test")
	print(Loss)		
	
	plt.plot(iter_rec, ls_rec)
	plt.xlabel("iter")
	plt.ylabel("Loss")
	plt.savefig("SGD.png")
	"""
	
	

	
	
if __name__ == "__main__":
	main()