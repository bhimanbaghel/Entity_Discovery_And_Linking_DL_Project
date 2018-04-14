'''#############################################################
Authors	: Bhiman Kumar Baghel	17CS60R74
			: Hussain Jagirdar 		17CS60R83
			: Lal Sridhar			17CS60R39
			: Nikhil Agarwal		17CS60R70
			: Shah Smit Ketankumar	17CS60R72
Usage		: Deep learning project (Entity Discovery and Linking)
Data		: 13-04-2018 
#############################################################'''


#Modules to be used
from bs4 import BeautifulSoup as BS
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import nltk
import pickle
import pysolr
import os
import io
import sys
import shutil

#Setting random seed
tf.set_random_seed(1234)

#Relative paths
data_path = "../dataset/source/"
weights_dir = "../weights/"

#loading input files
def get_source(file):
	fd = io.open(data_path+file,"r",encoding='utf-8')
	xml_file = fd.read()
	xml_soup = BS(xml_file, 'xml')
	return xml_soup

#loading pickle file containing word-mention pair
def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

#creating input_x and input_y
#returns x and y
def get_features():
	
	data_dict = load_obj("../ere_dict_new/_entity_dict_seperate")

	file_list = os.listdir(data_path)
	
	input_text = []
	
	#loading text from all files
	for file in file_list:
		xml_data = get_source(file)
		in_text = nltk.word_tokenize((xml_data.find('TEXT').get_text().replace(":"," ").replace("?"," ").replace("!"," ").replace("."," ").replace("("," ").replace(")"," ").replace("-"," ").replace("+"," ")).lower())
		for text in in_text:
			if text not in input_text:
				input_text.append(text)

	#searching glove vector from the solr index
	solr = pysolr.Solr('http://localhost:8983/solr/glove', timeout=10)
	
	input_x_dict = dict()
	
	#One hot vector
	y_map = dict()
	y_map["MEN"] = np.array([[1,0]])	
	y_map["OTH"] = np.array([[0,1]])
	
	y = list()
	x = list()

	for i in input_text:
		results = solr.search('id:'+i)
		
		#if word not found in index then continue
		if(len(results)==0):
			continue

		if i in data_dict:
			y.append(y_map['MEN'])
			a_flag = True
		
		else:
			y.append(y_map["OTH"])
			a_flag = True
		
		for r in results:
			xx = np.array(r["vector"].split(" "))
			xx = xx.astype(float)
			x.append(xx)

	return x,y

def RNN(x, weights, biases, timesteps,num_hidden):

	# Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
	x = tf.unstack(x, timesteps, 1)

	# Define a lstm cell with tensorflow
	lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

	# Get lstm cell output
	outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

	# Linear activation, using rnn inner loop last output
	return tf.matmul(outputs[-1], weights['out']) + biases['out']

#main code starts here
def main():
	
	in_dict = {}
	in_text = []

	#for testing
	if sys.argv[1] == "--test":
		
		data_fd = io.open("../dataset/data.txt", "r", encoding='utf-8')
		data_text = data_fd.read()
		data_fd.close()
		
		#removing stop words
		stop_words = set(stopwords.words('english'))
		in_text = nltk.word_tokenize((data_text.replace(":"," ").replace("?"," ").replace("!"," ").replace("."," ").replace("("," ").replace(")"," ").replace("-"," ").replace("+"," ")).lower())
		
		#searching in solr
		solr = pysolr.Solr('http://localhost:8983/solr/glove', timeout=10)
		in_text = [w for w in in_text if w not in stop_words]

		for t in in_text:
			results = solr.search('id:'+t)
			if len(results) == 0:
				continue
			else:
				if t not in in_dict:
					for r in results:
						in_dict[t] = np.asarray(r["vector"].split(" "), dtype=np.float32)

	#getting x and y
	x,y = get_features()

	#dividing data into training and test data
	temp = len(x)*80.0/100
	x_train = x[:int(temp)]
	x_test = x[int(temp):]
	y_train = y[:int(temp)]
	y_test = y[int(temp):]
	
	x = x_train
	y = y_train
	
	#hyperparameters
	learning_rate = 0.001
	training_steps = 100
	batch_size = 128
	display_step = 200

	#parameters
	num_input = len(x[0]) 
	timesteps = 1 
	num_hidden = 100
	num_classes = 2
	batch_size = 1
	learning_rate = 0.0001

	#placeholders for tensorflow
	X = tf.placeholder("float", [None, None, num_input])
	Y = tf.placeholder("float", [None,num_classes])

	#weights and biases
	weights = {
	    'out': tf.Variable(tf.random_normal([num_hidden,num_classes])),
	}

	biases = {
	    'out': tf.Variable(tf.random_normal([num_classes]))
	}

	#tensorflow code
	logits = RNN(X, weights, biases, timesteps,num_hidden)
	
	#prediction
	prediction = tf.nn.softmax(logits)
	pred_out = tf.argmax(prediction,1)
	
	#cross entropy loss
	loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
	
	#back_propogate
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(loss_op)

	#finding accuracy
	correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	init = tf.global_variables_initializer()
	
	#saving the weights
	saver = tf.train.Saver()
	with tf.Session() as sess:

		# Run the initializer
		sess.run(init)

		#if training
		if sys.argv[1] == "--train":
			res_fd = io.open("./train_log_"+str(training_steps)+".txt", "w", encoding='utf-8')
			x_axis = []
			y_axis_train_acc = []
			y_axis_loss = []
			y_axis_test_acc = []

			#epoches 
			for step in range(1, training_steps+1):
				temp_loss = 0
				acc = 0
				print "Epoche",step,
				res_fd.write("Epoche "+unicode(step)+" ")
				for i in range(len(x)):
					batch_x, batch_y = x[i],y[i]

					# Reshape data to get 28 seq of 28 elements
					batch_x = batch_x.reshape((batch_size, timesteps, num_input))
					
					# Run optimization op (backprop)
					sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
					loss, cp = sess.run([loss_op, pred_out],feed_dict={X: batch_x,Y: batch_y})
					temp_loss += loss
				x_axis.append(step)
				
				#Train accuracy
				batch_x = np.vstack(x_train).reshape((len(x_train), timesteps, num_input))
				acc_train = sess.run(accuracy, feed_dict={X: batch_x,Y: np.vstack(y_train)})
				y_axis_train_acc.append(acc_train)
				print "Train Accuracy", acc_train,
				res_fd.write("Train Accuracy : "+unicode(acc_train)+" ")
				
				#Test Accuracy
				batch_x = np.vstack(x_test).reshape((len(x_test), timesteps, num_input))
				acc_test = sess.run(accuracy, feed_dict={X: batch_x,Y: np.vstack(y_test)})
				y_axis_test_acc.append(acc_test)
				print "Test Accuracy", acc_test
				res_fd.write("Test Accuracy : "+unicode(acc_test)+"\n")

			res_fd.close()

			#Storing the weights
			if not os.path.exists(weights_dir):
				os.makedirs(weights_dir)
			weights_path = weights_dir+"Epoche_"+str(training_steps)+"/"
			if os.path.exists(weights_path):
				shutil.rmtree(weights_path)
				os.makedirs(weights_path)
			else:
				os.makedirs(weights_path)

			save_path = saver.save(sess, weights_path+"model.ckpt")
			print "Weights Saved at :",weights_path
			
			#Plotting the graph
			plt.scatter(x_axis, y_axis_train_acc, label = 'Train Accuracy')
			plt.scatter(x_axis, y_axis_test_acc, label = 'Test Accuracy')
			plt.xlabel('No. of Epoches')
			plt.ylabel('Accuracy')
			plt.title('Epoche VS Accuracy')
			print("Optimization Finished!")
			plt.legend()
			plt.savefig("acc_vs_epoch_graph_"+str(training_steps)+".png")
			plt.show()
		
		#If test
		elif sys.argv[1] == "--test":
			weights_path = weights_dir+"Epoche_"+str(training_steps)+"/"
			new_saver = tf.train.import_meta_graph(weights_path+'model.ckpt.meta')
			new_saver.restore(sess, tf.train.latest_checkpoint(weights_path))
			for t in in_text:
				if t in in_dict:
					batch_x = in_dict[t]
				else:
					print "No vector for :",t
					continue
				batch_x = batch_x.reshape((batch_size, timesteps, num_input))
				pred = sess.run(pred_out, feed_dict={X:batch_x})
				if pred == 0:
					print t, "MEN"
				elif pred == 1:
					print t, "OTH"


if __name__ == '__main__':
	main()