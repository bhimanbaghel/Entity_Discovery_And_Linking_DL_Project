'''#############################################################
Authors		: Bhiman Kumar Baghel	17CS60R74
			: Hussain Jagirdar 		17CS60R83
			: Lal Sridhar			17CS60R39
			: Nikhil Agarwal		17CS60R70
			: Shah Smit Ketankumar	17CS60R72
Usage		: parsing the source file
Data		: 13-04-2018 
#############################################################'''

from bs4 import BeautifulSoup as BS
import sys
import io
import os
import pickle
import shutil

reload(sys)
sys.setdefaultencoding("utf-8")

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_source():
	# entity_dict = load_obj("../ere_dict/ENG_NW_001278_20130115_F00013C4Y.rich_ere.xml_entity_dict")
	# print "Entity dictionary loaded"
	fd = io.open("../dataset/source/ENG_NW_001278_20130115_F00013C4Y.xml","r",encoding='utf-8')
	xml_file = fd.read()
	xml_soup = BS(xml_file, 'xml')
	return xml_soup
# data_str = str(xml_soup.get_text())
# data_str = data_str.strip('\n')
# print data_str.replace('\n', '')[101:106]
# print str(xml_soup.prettify())[101:]
xml_str = str(list(get_source())[0]) 
# print xml_str[253:260]
# print xml_soup.find('AUTHOR').get_text()
# xml_list = xml_soup.get_text()
# xml_str = str(xml_list[0])
# print xml_list
# file = open("./data.txt", "r")
# data = file.read()
# print data.replace("\n","")[109:114]

# for entity in entity_dict:
# 	e_type = entity_dict[entity]['entity_type']
# 	for entity_mention in entity_dict[entity]['mention_dict']:
# 		# print entity_mention
# 		em_offset = entity_dict[entity]['mention_dict'][entity_mention]['entity_mention_offset']
# 		# print em_offset
# 		em_length = entity_dict[entity]['mention_dict'][entity_mention]['entity_mention_length']
# 		em_noun_type = entity_dict[entity]['mention_dict'][entity_mention]['entity_mention_noun_type']
# 		em_text = entity_dict[entity]['mention_dict'][entity_mention]['entity_mention_text']
# 		if em_offset > 170:
# 			data_text = xml_str[em_offset+21:em_offset+21+em_length]
# 		else:
# 			data_text = xml_str[em_offset-3:em_offset-3+em_length]
		
# 		if em_text != data_text:
# 			print em_text, e_type, em_noun_type
# 			print "missmatch\n\n"
# 			# exit(0)
# 		else:
# 			print data_text
# 			print e_type
# 			print em_noun_type
# 			print "\n\n"