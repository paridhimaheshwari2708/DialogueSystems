import spacy
import codecs
import argparse
import googletrans
from SP import augmenter
from IO import conllud_sent
from transformers import PegasusForConditionalGeneration, PegasusTokenizerFast


class Paraphrase:
	def __init__(self):
		self.model = PegasusForConditionalGeneration.from_pretrained('tuner007/pegasus_paraphrase')
		self.tokenizer = PegasusTokenizerFast.from_pretrained('tuner007/pegasus_paraphrase')

	def get_paraphrased_sentences(self, sentence, num_return_sequences=5, num_beams=5):
		# tokenize the text to be form of a list of token IDs
		inputs = self.tokenizer([sentence], truncation=True, padding='longest', return_tensors='pt')
		# generate the paraphrased sentences
		outputs = self.model.generate(
			**inputs,
			num_beams=num_beams,
			num_return_sequences=num_return_sequences,
			)
		# decode the generated sentences using the tokenizer to get them back to text
		return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


class Translate:
	def __init__(self):
		self.translator = googletrans.Translator()
		print(f'All supported languages: {googletrans.LANGUAGES}')

	def get_translated_sentences(self, sentence):
		sentence_translated = self.translator.translate(sentence, src='en', dest='es').text # English to Spanish
		output = self.translator.translate(sentence_translated, src='es', dest='en').text # Spanish to English
		return output


class Crop_Rrotate:
	def __init__(self, maxrot=3,prob=1.0, loi = ["nsubj", "dobj", "iobj", "obj", "obl", "pobj"], 
					pl="root", multilabs = ["case", "fixed", "flat", "cop", "compound"]  ):
		self.nlp = spacy.load('en_core_web_sm')
		self.maxrot = maxrot
		self.prob = prob 
		self.loi = loi
		self.pl = pl
		# for predicate
		self.multilabs = multilabs 
		
	def get_augmentation(self, sentence, operation):
		sent = self.nlp(sentence)

		if operation == "rotate":
			ud_sent = conllud_sent.conllUDFromSent(sent).sent
			rotator = augmenter.rotator(ud_sent, aloi=self.loi, pl=self.pl, multilabs=self.multilabs, prob=self.prob)
			augSentRows, augSentTexts = rotator.rotate(maxshuffle=self.maxrot)
		elif operation == "crop":
			ud_sent = conllud_sent.conllUDFromSent(sent).sent
			cropper = augmenter.cropper(ud_sent, aloi=self.loi, pl=self.pl, multilabs=self.multilabs, prob= self.prob)
			augSentRows, augSentTexts = cropper.crop()

		# augSentTexts -- decreasing order sorting based on the length
		augSentTexts = sorted(augSentTexts, key=lambda x: (-len(x), x))

		# This will be None if empty/ no output found - augSentTexts
		return augSentTexts


# if __name__=='__main__':
# 	sentence = 'Learning is the process of acquiring new understanding, knowledge, behaviors, skills, values, attitudes, and preferences.'

# 	aug = Paraphrase()
# 	output = aug.get_paraphrased_sentences(sentence, num_return_sequences=10, num_beams=10)
# 	print(output)

# 	aug = Translate()
# 	output = aug.get_paraphrased_sentences(sentence, num_return_sequences=10)
# 	print(output)

# 	# sentence = 'Show me the booking from SF to LA '
# 	sentence = 'Yes, the Jesus green outdoor pool get the most consistently positive feedback'
# 	# sentence = 'can I help with anything else?'

# 	aug = Crop_Rrotate()
# 	output = aug.get_augmentation(sentence, operation='rotate')
# 	print(output)
# 	output = aug.get_augmentation(sentence,  operation='crop')
# 	print(output)
