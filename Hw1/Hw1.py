import numpy as np
import pandas as pd


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
	print(X)
	return X,Y

def GD(X, Y, lr, iter):
	theta = np.zeros([X.shape[1],1])
	Loss = np.sqrt(np.sum((Y-np.dot(X,theta))**2)/X.shape[0])
	print(Loss)
	for i in range(iter):
		gd_theta = -2* np.dot( X.transpose(), (Y-np.dot(X,theta)) ) #-2*X^T(Y-X*theta)
		theta = theta - lr * gd_theta
	return theta
	
			
def main():
	X,Y = FetchData(9)
	
	theta = GD(X, Y, 1e-7, 1000)
	Loss = np.sqrt(np.sum((Y-np.dot(X,theta))**2)/X.shape[0])
	
	result = np.dot(X,theta)
	cmp = np.concatenate((Y,result),axis=1)
	print(cmp)
	print(Loss)	
	
	
	
	
if __name__ == "__main__":
	main()