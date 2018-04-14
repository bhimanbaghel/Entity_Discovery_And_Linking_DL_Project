'''#############################################################
Authors	: Bhiman Kumar Baghel	17CS60R74
			: Hussain Jagirdar 		17CS60R83
			: Lal Sridhar			17CS60R39
			: Nikhil Agarwal		17CS60R70
			: Shah Smit Ketankumar	17CS60R72
Usage		: Deep learning project (Entity Discovery and Linking)
Data		: 13-04-2018 
#############################################################'''


****************************************************************
#1 Required packages and liberaries
****************************************************************
	1.	BeautifulSoup
	2.	Tensorflow
	3.	Solr
	4.	Glove Vectors
	5.	pysolr
	6.	numpy
	7.	matplotlib
	8.	nltk
	9.	stanford-corenlp-full-2018-02-27



****************************************************************
#2	Instructions to run
****************************************************************
	1.	Make sure you have all the file at correct place as shown in 		PIC.png file

	2.	Setting up Solr
		a.	Install Solr
		b.	Copy the './glove/' directory which is provided
			Navigate to Solr installation directory and then navigate to
			'./server/solr/'
			Paste the copied './glove/' directory here

	3.	Start Solr
		a.	Navigate to Solr installation directory
		b.	Open the terminal and run the following command:
			bin/solr start

	4.	Running the implementation code
		a.	Navigate to the './code/' directory which is provided
		
		b.	***Training the model***
			
			Note: 	Weights of pretrained models for 100 epoches and 5 			epoches are already provide in "./weights/" directory

					In mention_identification.py 
					At line no. 158
					training_steps are already set to 100 epoches
					you can change the no. of epoches by changing this value

			Open the terminal and run the following command:
			python mention_identification.py --train

			The model will be saved in the './weights/' directory

		c.	***Testing The model***

			Note :	Provide the data for testing in './dataset/' 				directory
					One sample test data is already provide as data.txt

			Open the terminal and run the following command:
			python mention_identification.py --test




******************************************************************
#3 Dataset
******************************************************************
	1.	We have provided the Rich ERE corpora dataset in './dataset/' directory
	
	2.	The dataset is already parsed and strored in the dictionary

	3.	If you have new dataset in the same format as Rich ERE corpora
		Then you can use the following programs in './code/' directory to parse the dataset:
		a.	ere_xml_parser.py
		b.	ere_xml_parser_1.py
		c.	source_xml_parser.py



*************************THANK YOU!!!************************************





