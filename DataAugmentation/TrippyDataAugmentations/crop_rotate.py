from IO import conllud_sent
from SP import augmenter
import codecs
import argparse
import spacy

class Crop_Rrotate:
    def __init__(self, maxrot=3,prob=1.0, loi = ["nsubj", "dobj", "iobj", "obj", "obl", "pobj"], 
                 pl="root", 
                 multilabs = ["case", "fixed", "flat", "cop", "compound"]  ):
        self.nlp = spacy.load('en_core_web_sm')
        self.maxrot = maxrot
        self.prob = prob 
        self.loi = loi
        self.pl = pl
        # for predicate
        self.multilabs = multilabs 
        
    def get_augmentation(self, sentence, operation):
        sent= self.nlp(sentence)
        
        if operation=="rotate":
            ud_sent = conllud_sent.conllUDFromSent(sent).sent
            rotator = augmenter.rotator(ud_sent, aloi=self.loi, pl=self.pl, multilabs=self.multilabs, prob=self.prob)
            augSentRows, augSentTexts = rotator.rotate(maxshuffle=self.maxrot)
                
        elif operation=="crop":
            # for sent in doc:
            ud_sent = conllud_sent.conllUDFromSent(sent).sent
            cropper = augmenter.cropper(ud_sent, aloi=self.loi, pl=self.pl, multilabs=self.multilabs, prob= self.prob)
            augSentRows, augSentTexts = cropper.crop()

        ## augSentTexts -- decreasing order sorting based on the length
        augSentTexts = sorted(augSentTexts, key=lambda x: (-len(x), x))

        ## This will be None if empty/ no output found - augSentTexts
        return augSentTexts
        
# if __name__=='__main__':
#     # sentence = 'Show me the booking from SF to LA '
#     # sentence = 'Show me the booking from SF to LA '
#     sentence = 'Yes, the Jesus green outdoor pool get the most consistently positive feedback'
#     # sentence = 'can I help with anything else?'

#     aug = crop_rotate()
#     output = aug.get_augmentation(sentence, operation='rotate')
#     print(output)

#     print("------")
#     output = aug.get_augmentation(sentence,  operation='crop')
#     print(output)    
