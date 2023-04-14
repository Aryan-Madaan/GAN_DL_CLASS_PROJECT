import numpy as np
file = open('list_attr_celeba.txt') 
attr =[]
attr_list = []
my_array = []
for line in file:
    '''
        Getting the attributes of the first image and storing them in a attr_list
    '''
    attr_list = line.split(' ')
    # attr_list = attr_list[:-1]
    break

def str2int(data):
    for i in range(0, len(data)):
        if data[i]=='1':
            data[i] = 1.0
        else:
            data[i] = 0.0
    return data

for line in file:
    attr = line.split(' ')
    my_array.append(str2int(attr[1:]))

np.save("trainX", my_array)
print(my_array)