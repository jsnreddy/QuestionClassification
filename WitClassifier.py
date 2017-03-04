#!/usr/bin/env python
"""
Wit.ai model training from the data
Details about wit model - https://wit.ai/blog and https://code.facebook.com/posts/181565595577955/introducing-deeptext-facebook-s-text-understanding-engine/  
Details about the API - https://wit.ai/docs/http/20160526#post--entities-link
"""

import urllib, urllib2, json, sys
from wit import Wit

import appConfig

def trainWitViaAPI():
	fname = appConfig.DATA_FOLDER + '/' + appConfig.TRAIN_FILE

	with open(fname) as f:
	    content = f.readlines()[1:]

	content = [x.strip().split('\t') for x in content] 

	trainData = {}

	for x in content:
		if x[2] in trainData.keys():
			trainData[x[2]].append(x[1])
		else:
			temp = []
			temp.append(x[1])
			trainData[x[2]] = temp

	values = []

	for x in trainData:
		value = {}
		value['value'] = x
		value['expressions'] = trainData[x]
		values.append(value)

	trainJSON = {}

	trainJSON['doc'] = 'Detect class of sentence'
	trainJSON['lookups'] = ['trait']
	trainJSON['id'] = 'intent'
	trainJSON['values'] = values

	url = 'https://api.wit.ai/entities?v=20160526'
	req = urllib2.Request(url)
	req.add_header('Authorization', 'Bearer IOUMYDGN3NPIPTZLCZITYRYVB6QXZ6T4')
	req.add_header('Content-Type', 'application/json')
	req.get_method = lambda: 'POST'

	try:
		response = urllib2.urlopen(req, json.dumps(trainJSON))
	except urllib2.HTTPError as e:
		print "Error occured while creating entity"
		print e.code
		print e.read()
		sys.exit(1)
		
client = Wit(access_token='IOUMYDGN3NPIPTZLCZITYRYVB6QXZ6T4')

response = client.message('How did serfdom develop in and then leave Russia ?')

# print('Wit.ai response: ' + str(response))

print('Text : ' + str(response['_text']))
print('Category : ' + str(response['entities']['intent'][0]['value']))
print('Confidence : ' + str(response['entities']['intent'][0]['confidence']))

