import re
import nltk
import spacy
import codecs
import argparse
import googletrans
from transformers import PegasusForConditionalGeneration, PegasusTokenizerFast

from SP import augmenter
from IO import conllud_sent


class Paraphrase:
	def __init__(self):
		self.model = PegasusForConditionalGeneration.from_pretrained('tuner007/pegasus_paraphrase').cuda()
		self.tokenizer = PegasusTokenizerFast.from_pretrained('tuner007/pegasus_paraphrase')
		self.model.eval()

	def get_paraphrased_sentences(self, sentence, num_return_sequences=5, num_beams=5):
		# Splitting into individual sentences
		sentence = re.split('[.?!]', sentence)
		sentence = [x.strip() for x in sentence]
		sentence = [x for x in sentence if x != '']

		# tokenize the text to be form of a list of token IDs
		inputs = self.tokenizer(sentence, truncation=True, padding='longest', return_tensors='pt').to('cuda:0')

		# generate the paraphrased sentences
		output = self.model.generate(
			**inputs,
			num_beams=num_beams,
			num_return_sequences=num_return_sequences,
			)

		# decode the generated sentences using the tokenizer to get them back to text
		output = self.tokenizer.batch_decode(output, skip_special_tokens=True)
		output = [output[i :: num_return_sequences] for i in range(num_return_sequences)]
		output = [' '.join(x) for x in output]
		return output


class Translate:
	def __init__(self):
		self.translator = googletrans.Translator()
		print(f'All supported languages: {googletrans.LANGUAGES}')

	def get_translated_sentences(self, sentence):
		sentence_translated = self.translator.translate(sentence, src='en', dest='es').text # English to Spanish
		output = self.translator.translate(sentence_translated, src='es', dest='en').text # Spanish to English
		return output


class Crop_Rotate:
	def __init__(self, maxrot=3,prob=1.0, loi = ['nsubj', 'dobj', 'iobj', 'obj', 'obl', 'pobj'], 
					pl='root', multilabs = ['case', 'fixed', 'flat', 'cop', 'compound']  ):
		self.nlp = spacy.load('en_core_web_sm')
		self.maxrot = maxrot
		self.prob = prob 
		self.loi = loi
		self.pl = pl
		# for predicate
		self.multilabs = multilabs 
		
	def get_augmentation(self, sentence, operation):
		# Splitting into individual sentences
		sentence_list = nltk.sent_tokenize(sentence)

		output = ''
		for sent_item in sentence_list:
			sent = self.nlp(sent_item)
			
			if operation == 'rotate':
				ud_sent = conllud_sent.conllUDFromSent(sent).sent
				rotator = augmenter.rotator(ud_sent, aloi=self.loi, pl=self.pl, multilabs=self.multilabs, prob=self.prob)
				augSentRows, augSentTexts = rotator.rotate(maxshuffle=self.maxrot)
					
			elif operation == 'crop':
				ud_sent = conllud_sent.conllUDFromSent(sent).sent
				cropper = augmenter.cropper(ud_sent, aloi=self.loi, pl=self.pl, multilabs=self.multilabs, prob= self.prob)
				augSentRows, augSentTexts = cropper.crop()

			# Decreasing order sorting based on the length
			augSentTexts = sorted(augSentTexts, key=lambda x: (-len(x), x))

			# None if empty / no output found
			if len(augSentTexts) != 0:
				op_sent = augSentTexts[0]
			else:
				op_sent = sent_item # If null, take the original sentence as output
			output += op_sent

		# Final output sentence (match the initial sentence where the function is called)
		return output


# if __name__=='__main__':
# 	sentences = [
# 		"Learning is the process of acquiring new understanding, knowledge, behaviors, skills, values, attitudes, and preferences.",
# 		"no , i just need to make sure it is cheap . oh , and i need parking",
# 		"hello , i have been robbed . can you please help me get in touch with the police ?",
# 		"was parkside the address of the police station ? if not , can i have the address please ?",
# 		"yes please . i also need the travel time , departure time , and price .",
# 		"After being struck by lightning and being affected by particle excelerator explosion, Barry Allen wakes up with incredible speed. He calls himself the flash. Now he is desperate to find the person that killed his mother when he was a child. Barry travels back in time on multiple occasions and screws everything up several times and ruins his friends lives but he's a funny guy. He is also a superhero and has saved hundreds of people's lives so he's a good guy. The flash continually gets help from other superheroes like the arrow and Supergirl",
# 		"Show me the booking from SF to LA ",
# 		"Yes, the Jesus green outdoor pool get the most consistently positive feedback",
# 		"can I help with anything else?",
# 	]

# 	aug = Paraphrase()
# 	for sent in sentences:
# 		output = aug.get_paraphrased_sentences(sent, num_return_sequences=2, num_beams=10)
# 		print(output)

# 	aug = Translate()
# 	for sent in sentences:
# 		output = aug.get_translated_sentences(sent)
# 		print(output)

# 	aug = Crop_Rotate()
# 	for sent in sentences:
# 		output = aug.get_augmentation(sent, operation='rotate')
# 		print(output)
# 		output = aug.get_augmentation(sent,  operation='crop')
# 		print(output)
