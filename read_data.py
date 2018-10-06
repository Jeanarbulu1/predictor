import os,glob
from word import embeddings_index
from collections import Counter
import soundfile as sf
import numpy as np
path = '/content/drive/My Drive/train-clean-100.tar/LibriSpeech/train-clean-100'

direc= os.listdir(path)

labels = []					
def generator(batch_size):
	batch_features = np.zeros((1,2,1))
	try:
		for i in direc:
			test  = os.path.join(path,i)
			direc2 = os.listdir(test)
			for j in direc2:
				test2 = os.path.join(test,j)
				direc3 = os.listdir(test2)
				for i,x in enumerate(direc3):
					test3 = os.path.join(test2,x)
					if x.endswith(".flac"):
						while i < batch_size:
							y = sf.read(test3)
							z = np.array(y)
							z = z.reshape(1,2,1)
							np.append(batch_features,z)
							break

					else:
						y = open(test3)
						label = [line.split(',') for line in y]
						a = []
						for i in label:
							for j in i:
								j = j.strip('\n')
								y = ' '.join(j.split(' ')[1:])
								a.append(y)
						l = []
						for i in a:
							i.lower()
							l.append(i)
						vocab = Counter(w for txt in l for w in txt.split())
						matrix_len = len(vocab)
						weights_matrix = np.zeros((batch_size,matrix_len, 100))
						words_found = 0
						for i, word in enumerate(vocab):
							while i < batch_size:
								try: 
									weights_matrix[i] = embeddings_index[word]
									words_found += 1
							
								except KeyError:
									weights_matrix[i] = np.random.normal(scale=0.6, size=(100, ))
										
							
	except MemoryError:
		print(MemoryError)

	yield batch_features,weights_matrix



#set vectors of sound features


#Audio feature
