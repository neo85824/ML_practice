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
	print(X)
	print(Y)
			
def main():
	FetchData(9)
	
if __name__ == "__main__":
	main()