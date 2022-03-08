import googletrans
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

    # if __name__=='__main__':
    # sentence = 'Learning is the process of acquiring new understanding, knowledge, behaviors, skills, values, attitudes, and preferences.'

    # aug = Paraphrase()
    # output = aug.get_paraphrased_sentences(sentence, num_return_sequences=10, num_beams=10)
    # print(output)

    # aug = Translate()
    # output = aug.get_paraphrased_sentences(sentence, num_return_sequences=10)
    # print(output)