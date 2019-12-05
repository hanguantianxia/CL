import re
from typing import List, Dict
import os
import json








def process_doc(doc:str, min_len:int=8,delete_English=True ,delete_punctuation=True) -> List[str]:
	"""
	preprocess the doc text from the regex , it will finish following things:
		1. delete the empty lines
		2. delete the short lines whose length is shorter than min_len
		3. delete the spaces
		4. delete the punctuation by the bool
		5. return the list of sentence processed by 1,2

	:param doc: the doc string with empty lines and short lines
	:param min_len: the min length of line
	:param delete_punctuation: bool to determine if delete the punctuations in the sentence
	:return: the List of sentence processed by 1,2
	"""
	### 1. delete the empty lines

	pattern = re.compile("\n\n+", re.S)
	pattern1 = re.compile(" ", re.S)
	doc_no_empyt_line = re.sub(pattern, "\n" , doc)
	doc_no_empyt_line = re.sub(pattern1, "", doc_no_empyt_line)

	# print(doc_no_empyt_line) # output the document without empty line

	if delete_English is True:
		pattern2 = re.compile("[a-zA-z]")
		doc_no_empyt_line = pattern2.sub("", doc_no_empyt_line)

	if delete_punctuation is True:
		# pattern2 = re.compile(""""[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）"]+""")
		pattern3 = re.compile( r',|/|：|;|:|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|、|‘|’|【|】|·|！|”|“| |…|（|）|」|「|《|》', re.S)
		doc_no_empyt_line = pattern3.sub(" ", doc_no_empyt_line)
		pattern_period = re.compile("\.|。|;|；", re.S)
		doc_no_empyt_line = pattern_period.sub("\n", doc_no_empyt_line)
		doc_no_empyt_line = re.sub(pattern, "\n", doc_no_empyt_line)
		pattern4 = re.compile(" +", re.S)
		doc_no_empyt_line = pattern4.sub(" ", doc_no_empyt_line)

	# print(doc_no_empyt_line)
	raw_sentence_list = doc_no_empyt_line.split("\n")
	# print(len(raw_sentence_list))
	sentence_list = []
	# i = 0
	for line in raw_sentence_list:
		if len(line) > min_len:
			sentence_list.append(line)
			# i = i + 1
			# if i % 1000 == 1:
			# 	print("Now have finish {} lines".format(i))

	return  sentence_list



def get_doc(filename :str) -> List[List[str]]:
	"""
	from the file preprocessed by 
	:param filename: the txt file preprocessed by cht2chs and the type is <doc>....</doc>
	:return:List[List[str]] the outside list is for the documents, the inside List is for lines in document
	"""
	data = []
	try:
		with open(filename, 'r', encoding='utf-8') as f:
			content = f.read()
			# print(content)
			pattern = re.compile(r"<doc.*?>(.*?)</doc>",re.S)
			texts = re.findall(pattern, content)
			# print(data)

			for text in texts:
				# print(text)
				temp = process_doc(text)
				data.extend(temp)
				# print(len(temp))

			return data

	except IOError as e:
		print("the file {} cannot open".format(filename))
		print(e)
		raise IOError


def segment(raw_sents:List[str], segment="jieba") -> List[List[str]]:
	"""
	segment the Chinese sentence by pkuseg package

	:param raw_sent:

	:return:
	"""
	# segment_list = ["pkuseg", "jieba"]
	# if segment.strip() not in segment_list:
	# 	return []

	seg_sents = []
	if segment == "pkuseg":
		import pkuseg

		## init the seg
		seg = pkuseg.pkuseg()

		## segment the sentence by pkuseg
		for sent in raw_sents:
			res_seg = seg.cut(sent)
			seg_sents.append(res_seg)
		# print(seg_sents)
	elif segment == "jieba":
		import jieba
		for sent in raw_sents:
			res_seg = jieba.lcut(sent)
			sentence = " ".join(res_seg)
			pattern4 = re.compile(" +", re.S)
			sentence = pattern4.sub(" ", sentence)
			res_seg = sentence.split(" ")
			seg_sents.append(res_seg)

	return seg_sents




def merge_dict(dict_A:Dict, dict_B:Dict) -> Dict:
	"""
	merge 2 dictionary
	:param dict_A:
	:param dict_B:
	:return: a dictionary merged by dictionary A and B
	"""
	merged_dict = dict_A.copy()
	for key in dict_B.keys():
		merged_dict[key] = merged_dict.get(key, 0) + dict_B.get(key)

	return merged_dict

def get_lexicon(seg_sents:List[List[str]]) -> Dict[str, int]:
	"""
	to build a lexicon and count the times every word occurs in the sentence

	:param seg_sents: the sentence
	:param max_lexicon_lenth: the max lexicon_length
	:return: the dictionary mapping from word to int, represent the times every word occurs in the corpus
	"""
	lexicon = {}

	for sent in seg_sents:
		for word in sent:
			lexicon[word] = lexicon.get(word, 0) + 1
	# print(lexicon)
	return lexicon


def save_lexicon(lexicon:Dict[str, int], filename:str):
	"""

	:param lexicon:
	:return:
	"""
	with open(filename, 'w', encoding="utf-8") as f:
		json_str = json.dumps(lexicon)
		f.write(json_str)


def read_lexicon(filename) -> Dict:
	"""

	:param filename:
	:return:
	"""
	lexicon = None
	with open(filename, 'r', encoding="utf-8") as f:
		lexicon = json.load(f)
	return  lexicon

def save_list(filename:str, seg_sents:List[List[str]]):
	"""
	save the segmented document
	:param filename:
	:param data:
	"""
	with open(filename, 'w', encoding="utf-8") as f:
		for sent in seg_sents:
			sentence = " ".join(sent)
			# print(sentence)
			f.write(sentence + '\n')





def main():
	corpus_path = os.path.join(".", "corpus_200M")
	corpus_list = [os.path.join(corpus_path,f) for f in os.listdir(corpus_path)
				   if os.path.isfile(os.path.join(corpus_path,f))]

	lexicons = {}
	for file in corpus_list:
		data = get_doc(file)
		seg_sents = segment(data)
		save_list(file + "_seg", seg_sents)
		lexicon = get_lexicon(seg_sents)
		lexicons = merge_dict(lexicons, lexicon)

	save_lexicon(lexicons, "lexicon.json")





if __name__ == '__main__':
	# data = get_doc(r"E:\Program\CL\corpus_200M\wiki_00")
	# seg_sents = segment(data)
	# lexicon = get_lexicon(seg_sents)

	main()
	# the len should be 3m