# https://cntk.ai/pythondocs/CNTK_202_Language_Understanding.html
# CNTK 202: Language Understanding with Recurrent Networks
# Author: wuyi
# Date : 2018-03-36

"""
This tutorial shows how to implement a recurrent network to process text, 
for the Air Travel Information Services (ATIS) task of slot tagging 
(tag individual words to their respective classes, 
where the classes are provided as labels in the training data set).
There are 2 parts to this tutorial: 
	- Part 1: We will tag each word in a sequence to their corresponding label 
	- Part 2: We will classify a sequence to its corresponding intent.
"""

from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)
import requests
import os

# ------ 1. Data ------
filepath = 'E:/ProgramLib/Python/CNTK/testdata/CNTK202/'
def download(url, filename):
	""" utility function to download a file"""
	response = requests.get(url, stream=True)
	with open(filename, "wb") as handle:
		print(filename)
		for data in response.iter_content():
			handle.write(data)

locations = ['Tutorials/SLUHandsOn', 'Examples/LanguageUnderstanding/ATIS/BrainScript']

data = {
    'train': {'file': 'atis.train.ctf', 'location': 0},
	'test': {'file': 'atis.test.ctf', 'location': 0},
	'query': {'file': 'query.wl', 'location': 1},
	'slots': {'file': 'slot.wl', 'location': 1},
	'intent': {'file': 'intent.wl', 'location': 1}
}

isDownload = False
if isDownload :
	for item in data.values():
		location = locations[item['location']]
		path = filepath + location + '/' + item['file']
		print(path)
		if os.path.exists(path):
			print("Reusing locally cached:", item['file'])
			# Update path
			item['file'] = path
		elif os.path.exists(item['file']):
			print("Reusing locally cached:", item['file'])
		else:
			print("Starting download:", item['file'])
			url = "https://github.com/Microsoft/CNTK/blob/release/2.5/%s/%s?raw=true"%(location, item['file'])
        download(url, item['file'])
        print("Download completed")


