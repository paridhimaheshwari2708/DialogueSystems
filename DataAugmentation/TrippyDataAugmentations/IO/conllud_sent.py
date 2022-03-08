# encoding: utf-8

"""
Created by Gözde Gül Şahin
20.05.2018
Read UD treebanks into a data structure suitable to cropping and flipping
"""

__author__ = 'Gözde Gül Şahin'

import codecs
import copy
import imp

class conllUDFromSent:

    def __init__(self, sentence=None):
        """
        init: read sentences into conllUDsent structure
        """
        self.sent = []
        if sentence is not None:
            self.sent = self._parse_sent(sentence)
        else:
            print("No sentence given, check again!")
        

    def _parse_sent(self, sent):
        """
        read_file: read tokens and create a dependency tree for all UD sentences
        """
        
        # conllsentences = []
        # sent_no = 0

        csent = conllUDsent()
        rows = []
        for idx, word in enumerate(sent):
            if word.dep_.lower().strip() == "root":
                head_idx = 0
            else:
                head_idx = word.head.i + 1 - word.sent[0].i
            
            row_details = [idx+1, word.text, word.lemma_, word.pos_, word.tag_, '_', head_idx,word.dep_.lower(), '_' ]
            
            # import pdb
            # pdb.set_trace()
            # row_details = [idx+1] + [str(text, 'utf-8') for text in row_details_text]

            # ctoken = conllUDtoken(word,idx+1, head_idx)
            ctoken = conllUDtoken(row_details,idx+1 )
            csent.add_token(ctoken)
            
            rows.append(row_details)

        csent.tokenWords = [row[1] for row in rows]
        csent.tokenLemmas = [row[2] for row in rows]
        csent.rows = rows
        # conllsentences.append(csent)

        # build the dependency tree
        # head -> all children tokens
        for tok in csent.tokens:
            if tok.head in csent.deptree:
                csent.deptree[tok.head].append(tok)
            else:
                csent.deptree[tok.head] = [tok]
        # sent_no+=1

        return csent
        

        # sentences = strIn.split("\n\n")
        # sentences = strIn.split("\n")
        for sent in sentences:
            if(len(sent)>0):
                lines = sent.split("\n")
                # create new conllud sentence
                csent = conllUDsent()
                rows = []
                tok_index = 1
                for line in lines:
                    # create UD token (if it is not a comment line or an IG token)
                    if not line.startswith(u"#") and not (u"-" in line.split("\t")[0]):
                        # print(line.split('       '))
                        # ctoken = conllUDtoken(line.split("\t"),tok_index)
                        tokens = line.split()
                        ctoken = conllUDtoken(tokens,tok_index)
                        csent.add_token(ctoken)
                        rows.append(line.split())
                        tok_index+=1

                csent.rows = rows

                # add token words
                csent.tokenWords = [row[1] for row in rows]
                csent.tokenLemmas = [row[2] for row in rows]
                conllsentences.append(csent)

                # build the dependency tree
                # head -> all children tokens
                for tok in csent.tokens:
                    if tok.head in csent.deptree:
                        csent.deptree[tok.head].append(tok)
                    else:
                        csent.deptree[tok.head] = [tok]
            sent_no+=1
        return conllsentences

class conllUDsent:
    def __init__(self):
        self.tokens = []
        # token words and lemmas
        self.tokenWords = []
        self.tokenLemmas = []
        # for reordering purposes
        self.deptree = {}
        self.rows = []

    def add_token(self, token):
        self.tokens.append(token)

    def print_sent_ord(self,ord):
        """
        print_sent_ord: print sentence as text with the given order
        """
        strsent = ""
        for i in ord:
            if i==0:
                continue
            tok = self.tokens[i-1]
            strsent+=(tok.word+' ')
        return strsent

    def reorder(self, neword):
        """
        reorder: Reorder the tokens of the sentence with the given new order (neword)
        """
        newrows = []

        # mapping[oldorder]=neworder
        mapping = {}
        mapping[u"0"]=u"0"

        # first put them together
        # print(neword, len(neword), self.rows)
        
        for i, j in zip(neword,range(len(neword))):
            if i==0:
                continue
            # print(i, j)
            newrows.append(copy.deepcopy(self.rows[i-1]))
            mapping[self.tokens[i-1].depid] = self.tokens[j-1].depid

        # replace old ids with new ids
        for r in newrows:
            # change id
            r[0] = mapping[r[0]]
            if r[6] in mapping:
                r[6] = mapping[r[6]]
            else:
                r[6] = u"_"
        return newrows

class conllUDtoken:
    # def __init__(self, word, tok_index, head_idx):
    #     # index in the sentence (the order)
    #     self.index = tok_index
    #     # print(fields, len(fields))
    #     # dependency id for the dependency tree
    #     self.depid = tok_index
    #     self.word = word.text
    #     self.lemma = word.lemma
    #     self.pos = word.pos_
    #     self.ppos = word.tag_
    #     self.feat = '_'
    #     self.head = head_idx
    #     self.deprel = word.dep_.lower()
    #     self.pdeprel = '_'
    #     #######

    def __init__(self, fields, tok_index):
        # index in the sentence (the order)
        self.index = tok_index
        # print(fields, len(fields))
        # dependency id for the dependency tree
        self.depid = fields[0]
        self.word = fields[1]
        self.lemma = fields[2]
        self.pos = fields[3]
        self.ppos = fields[4]
        self.feat = fields[5]
        self.head = fields[6]
        self.deprel = fields[7]
        # self.pdeprel = fields[8]


        # if word.head is word:
        #     head_idx = 0
        # else:
        #      head_idx = word.i-sent[0].i+1
        # print(
        #     i+1, # There's a word.i attr that's position in *doc*
        #     word.pos_, # Coarse-grained tag
        #     word.tag_, # Fine-grained tag
        #     head_idx,
        #     word.dep_, # Relation
        #     '_', '_')
