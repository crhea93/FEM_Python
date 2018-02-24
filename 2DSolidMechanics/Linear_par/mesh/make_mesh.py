#Super basic making of structured mesh numbering starting at 1
import numpy as np
num_nodes = 5 #per line
num_elements = num_nodes - 1 #per line
x = np.linspace(0,1,num_nodes)
y = np.linspace(0,1,num_nodes)

file = open('../outputs/basic.txt','w')
count = 0
for i in range(num_nodes):
    for j in range(num_nodes):
        if count != 0:
            file.write('\n')
        file.write(str(x[j]))
        file.write(" ")
        file.write(str(y[i]))
        count += 1
file.close()

#now can we make a nice elemental connectivity matrix?
nodes = np.arange(1,num_nodes**2+1)
file = open('../outputs/basic_con.txt','w')
count = 0
for i in range(num_nodes+1,len(nodes)):
    if count !=num_elements:
        file.write(str(nodes[i-(num_nodes+1)]))
        file.write(" ")
        file.write(str(nodes[i-(num_nodes)]))
        file.write(" ")
        file.write(str(nodes[i]))
        file.write(" ")
        file.write(str(nodes[i-1]))
        file.write('\n')
        count += 1
    else:
        count = 0
file.close()

#now for the sides!
top = np.arange(1+num_nodes*num_nodes-num_nodes,num_nodes**2+1)
left = np.zeros(num_nodes,dtype=int)
bottom = np.arange(1,num_nodes+1)
right = np.zeros(num_nodes,dtype=int)
for i in range(len(left)):
    left[i] = 1+i*num_nodes
    right[i] = num_nodes*(i+1)


#print(top)
#print(left)
#print(bottom)
#print(right)

file = open('../outputs/basic_sides.txt','w')
for i in range(num_nodes):
    file.write(str(top[i]))
    file.write(" ")
    file.write(str(left[i]))
    file.write(" ")
    file.write(str(bottom[i]))
    file.write(" ")
    file.write(str(right[i]))
    file.write('\n')

file.close()
