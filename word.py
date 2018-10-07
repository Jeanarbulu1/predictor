from collections import Counter
import numpy as np

#Get GLOVE
try:
	embeddings_index = dict()
	f = open('/content/drive/My Drive/glove.6B.100d.txt',encoding = 'utf8')
	for line in f:
	    values = line.split()
	    word = values[0]
	    coefs = np.asarray(values[1:], dtype='float32')
	    embeddings_index[word] = coefs
	f.close()

except e:
	print(e)

