import sys

path = str(sys.argv[1])  #argument
file = open(path, 'r') 
data = file.read()
num = 0
space = " "   
ws = ""  #for writing files
list = data.split(" ")
record = [""]
for s in list:
	if record.count(s) == 0:
		l = [s, str(num), str(list.count(s))]  #count how mamy times the string appear
		ws = ws + space.join(l) + "\n"
		record.append(s)
		num += 1
		
file2 = open("Q2.txt",'w')
file2.write(ws)
		

		



		
		


		