
import tensorflow as tf 
import numpy as np 
from collections import Counter
import glob
import config
import language_map
import os
import pickle
import _preprocess
import preprocess
import sys


# 4 hidden layers DNN
# 6900 * 1000 * 500 * 200 * 100 * 4

num_classes = 4
num_epoch = 1000
learning_rate = 0.0008
dropout = 0.6
gram_size = 3
split_size = 15


class docClassifier:
	def __init__(self,rebuild):
		self.rebuild = rebuild

		if rebuild == False and os.path.isfile('unique_ngrams.p') :
			self.unique_ngrams = pickle.load(open('unique_ngrams.p','rb'))

		else :
			assert os.path.isdir(config.grams_dir)
			gramfiles = glob.glob(config.grams_dir+'/*')
			ngrams = []
			for (i,f) in enumerate(gramfiles) :
 				file = open(f,'r',encoding='ISO-8859-1')
 				ng = file.read().split()
 				ngrams.extend(ng)
		
			self.unique_ngrams = list(set(ngrams))
			pickle.dump(self.unique_ngrams,open('unique_ngrams.p','wb'))
			# release memory
			del ngrams
	
		self.num_features = len(self.unique_ngrams)
		self.num_classes = num_classes
		self.sess = tf.Session()

		self.keep_prob = tf.placeholder(tf.float32)
		self.weights = {
			'w1': tf.Variable(tf.random_normal([self.num_features, 1000], dtype = 'float32')),
			'w2': tf.Variable(tf.random_normal([1000, 500], dtype = 'float32')),
			'w3': tf.Variable(tf.random_normal([500, 200], dtype = 'float32')),
			'w4': tf.Variable(tf.random_normal([200, 100], dtype = 'float32')),
			'w5': tf.Variable(tf.random_normal([100, self.num_classes], dtype = 'float32'))
		}

		self.biases = {
			'b1' : tf.Variable(tf.random_normal([1000], dtype='float32')),
			'b2' : tf.Variable(tf.random_normal([500], dtype='float32')),
			'b3' : tf.Variable(tf.random_normal([200], dtype='float32')),
			'b4' : tf.Variable(tf.random_normal([100], dtype='float32')),
			'b5' : tf.Variable(tf.random_normal([self.num_classes], dtype='float32'))
		}

		self.X = tf.placeholder(tf.float32,[None,self.num_features])
		self.Y = tf.placeholder(tf.float32,[None,self.num_classes])


	def grams_to_vector(self,grams):
		vec = np.zeros((len(self.unique_ngrams),))
		count = Counter(grams)

		for (i,g) in enumerate(self.unique_ngrams) : 
			if g in grams:
				vec[i] = count[g]
	
		tot = np.sum(vec)
		vec /= tot
		return vec

	def MLP(self,predict=False):
		layer1_out = tf.add(tf.matmul(self.X, self.weights['w1']),self.biases['b1'])
		layer1_a = tf.nn.relu(layer1_out)
		layer2_out = tf.add(tf.matmul(layer1_a, self.weights['w2']),self.biases['b2'])
		layer2_a = tf.nn.relu(layer2_out)
		layer3_out = tf.add(tf.matmul(layer2_a, self.weights['w3']),self.biases['b3'])
		layer3_a = tf.nn.relu(layer3_out)
		layer4_out = tf.add(tf.matmul(layer3_a, self.weights['w4']),self.biases['b4'])
		layer4_a = tf.nn.relu(layer4_out)
		layer5_dropout = tf.nn.dropout(layer4_a, self.keep_prob)
		layer5_out = tf.add(tf.matmul(layer5_dropout, self.weights['w5']),self.biases['b5'])
		# layer5_a = tf.nn.relu(layer5_out)
		pred = layer5_out 

		if predict :
			return pred

		cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y,logits = pred))
		minimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

		return pred, cost, minimizer


	def initVariables(self):
		init = tf.global_variables_initializer()
		self.sess.run(init)


	def makeTrainSet(self):
		"""
		train files should be in train folder and the names of files should correspond to languageName_number
		returns feature vector and label
		"""
		if self.rebuild == False and os.path.isfile('train_x.p') and os.path.isfile('train_y.p'):
			x = pickle.load(open('train_x.p','rb'))
			y = pickle.load(open('train_y.p','rb'))
			return x,y 

		gl = glob.glob(config.train_dir+'/*')
		fnames = [ f.split('/')[1].split('_')[0] for f in gl]

		fnames = [ language_map.lang_map[lang] for lang in fnames ]

		labels = np.array(fnames,dtype=np.int)

		# convert to vector
		x = np.empty([len(gl), self.num_features])
		# one-hot encoding
		y = np.zeros([len(gl),num_classes])

		for (i,file) in enumerate(gl) :
			print('constructing feature vector from file {}'.format(file))
			fs = open(file,'r',encoding='ISO-8859-1')
			grams = fs.read().split()
			x[i] = self.grams_to_vector(grams)
			y[i][labels[i]] = 1.0
			fs.close()

		pickle.dump(x,open('train_x.p','wb'))
		pickle.dump(y,open('train_y.p','wb'))

		return x,y

	def makeFeatureVector(self):
		g = glob.glob(config.test_dir+'/*')
		for f in g:
			fs = open(f,'r',encoding='ISO-8859-1')
			text = fs.read(10000)
			if not os.path.isdir(config.test_language_dir):
				os.mkdir(config.test_language_dir)
			ft = open(config.test_language_dir+'/'+f.split('/')[-1],'w',encoding='ISO-8859-1')
			ft.write(text)
			fs.close()
			ft.close()

		preprocessor = _preprocess.preprocess(config.language_dir,gram_size)
		preprocessor._toLower(config.test_language_dir,config.test_lower_dir)._stem(config.test_lower_dir,config.test_stem_dir)._makeNgrams(config.test_stem_dir,config.test_grams_dir)

		gl = glob.glob(config.test_grams_dir+'/*')

		# convert to vector
		x = np.empty([len(gl), self.num_features])
		filename = [None]*len(gl)

		for (i,file) in enumerate(gl) :
			print('constructing feature vector from file {}'.format(file))
			fs = open(file,'r',encoding='ISO-8859-1')
			grams = fs.read().split()
			x[i] = self.grams_to_vector(grams)
			filename[i] = file.split('/')[-1]
			fs.close()

		return x,filename

	def trainAndValidate(self,retrain=True,r=0.8, n_epoch=num_epoch):
		if retrain == False:
			self.initVariables()
			self.restore()
			return

		x,y = self.makeTrainSet()
		n = np.arange(x.shape[0],dtype=int)
		np.random.shuffle(n)

		x = x[n]
		y = y[n]

		last = int(r * len(x))
		x_train = x[:last]
		y_train = y[:last]

		x_test = x[last:]
		y_test = y[last:]

		pred,cost,minimizer = self.MLP()
		self.initVariables()

		for i in range(1,n_epoch+1):
			_,c = self.sess.run([minimizer,cost],{self.X: x_train, self.Y: y_train, self.keep_prob: dropout})
			if i % 10 == 0:
				print(" epoch = {} cost = {} ".format(i,c))

		self.save()
		p = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,axis=1),tf.argmax(self.Y,axis=1)),dtype=tf.float32))
		p = self.sess.run(p,{self.X: x_test, self.Y: y_test,self.keep_prob: 1.0})
		print("Accuracy on test set = {} %".format(p*100))

	def predict(self):
		x,filename = self.makeFeatureVector()
		pred = self.MLP(predict=True)
		norm_pred = tf.divide(pred,tf.abs(tf.reduce_sum(pred,axis=1,keepdims=True)))
		acc = tf.nn.softmax(norm_pred)
		p = tf.argmax(pred,axis=1)
		p,acc = self.sess.run([p,acc],{self.X : x, self.keep_prob: 1.0})

		reverse_map = [ key for key in language_map.lang_map ]
		for i in range(len(p)):
			conf = acc[i][p[i]]*100
			print('{:<12}=>{:>12}  confidence : {:.2f} %'.format(filename[i],reverse_map[p[i]],conf))

	def save(self):
		saver = tf.train.Saver()
		path = saver.save(self.sess,"./model.ckpt")
		print("Model saved as ",path)

	def restore(self):
		f = glob.glob("model.ckpt*")
		if len(f) < 2:
			print("No saved model. Try rebuilding or retraining ")
			exit()
		saver = tf.train.Saver()
		saver.restore(self.sess,"./model.ckpt")
		print("Model restored...")
			

def showUsage():
	print("Usage :")
	print("python main.py    : run WITHOUT rebuilding or retraining all the files (uses already built features)")
	print("python main.py -r : run rebuilding all the files")
	print("python main.py -t : retrain model")

if len(sys.argv) > 2:
	showUsage()
	exit()

rebuild = False
retrain = False

if len(sys.argv) == 2:
	if sys.argv[1] == '-r':
		print("Rebuilding and training...")
		rebuild = True
	elif sys.argv[1] == '-t':
		print("Retraining model...")
		retrain = True
	else:
		print("Wrong parameter")
		showUsage()
		exit()

retrain = retrain or rebuild

preprocess.run(rebuild,gram_size,split_size)
dc = docClassifier(rebuild)
dc.trainAndValidate(retrain)

dc.predict()
