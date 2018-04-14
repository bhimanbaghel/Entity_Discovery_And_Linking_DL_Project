'''-----------Usage-----------
Command line Argument 1: Path to glove_data_set.txt
Command line Argument 2: SOLR PORT NUMBER
Command line Argument 3: CORE NAME
'''

import simplejson as json
import requests
from nltk.util import ngrams
from tokenizer import tokenizer
import sys
import os
PORT_NO=sys.argv[2]
CORE_NAME=sys.argv[3]
#to disable proxy for this program
for k in list(os.environ.keys()):
    if k.lower().endswith('_proxy'):
        del os.environ[k]
def update_bigram_solr_field(payload):
    base_url = 'http://localhost:'+PORT_NO+'/'
    solr_url = 'solr/'+CORE_NAME+'/'
    update_url = 'update?commit=true'
    full_url = base_url + solr_url + update_url
    headers = {'content-type': "application/json"}

    response = requests.post(full_url, data=json.dumps(payload), headers=headers)

    return response


def main():
    glove_file=open(sys.argv[1],"r").readlines()
    i = 0
    temp=[]
    for line in glove_file:
        line = line.strip() #stripping new lines
        line = line.split(" ",1)
        temp.append({"id": line[0], "vector": line[1]})
        i += 1
        if i%10000 == 0:
            print i,"processed"
            response = update_bigram_solr_field(temp)
            try:
                    response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                    print "Error: " + str(e)
            temp =[]
if __name__ == "__main__":
    main()
 
