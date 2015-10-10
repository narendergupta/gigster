from collections import defaultdict
from datetime import datetime
from gen_utils import *
from gigs.config.strings import *
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import csv
import json
import math
import statistics as stats

class DataModel:
    """Class for reading and managing raw data"""
    def __init__(self, gigsfile, chatfile):
        self.gigsfile = gigsfile
        self.chatfile = chatfile
    #enddef


    def read_data(self):
        self.read_gigs()
        self.read_chats()
    #enddef


    def read_gigs(self):
        gigs = {}
        with open(self.gigsfile) as gigs_f:
            gigreader = csv.DictReader(gigs_f)
            for row in gigreader:
                gig = self.__normalize_gig(row)
                gig[CHATS] = {}
                gigs[gig[ID]] = gig
        self.gigs = gigs
    #enddef


    def __normalize_gig(self, gig):
        json_features = [PLATFORMS, DOCS, KEY_FEATURES]
        for feature in json_features:
            if gig[feature] != '':
                gig[feature] = json.loads(gig[feature])
            else:
                gig[feature] = []
        #endfor
        gig[ID] = gig[ID].split('(')[1].split(')')[0]
        gig[PRICE] = int(float(gig[PRICE]))
        gig[CREATED] = datetime.strptime(gig[CREATED],'%Y-%m-%dT%H:%M:%S.%fZ')
        gig[START_DATE] = datetime.strptime(gig[START_DATE],'%Y-%m-%dT%H:%M:%S.%fZ')
        if 'true' in gig[STALE]:
            gig[STALE] = True
        elif 'false' in gig[STALE]:
            gig[STALE] = False
        else:
            gig[STALE] = None
        return gig
    #enddef


    def read_chats(self):
        chats = []
        with open(self.chatfile) as chats_f:
            json_data = json.load(chats_f)
            self.chats = json_data[MESSAGES]
            for key in self.chats.keys():
                if key in self.gigs:
                    self.gigs[key][CHATS] = self.chats[key]
                #endif
            #endfor
    #enddef


    def process_docs(self):
        types = []
        subtypes = []
        for key in self.gigs.keys():
            gig = self.gigs[key]
            for doc in gig[DOCS]:
                types.append(doc[TYPE])
                subtypes.append(doc[SUBTYPE])
        print(unique(types))
        print(unique(subtypes))
        return None
    #enddef


    def __make_zero_mean_one_std(self, vals):
        mean = stats.mean(vals)
        std = stats.stdev(vals)
        vals = [float(val-mean)/std for val in vals]
        return vals
    #enddef


    def __get_doc_feature_labels(self):
        types = [SYSTEM, ATTACHMENT]
        # intentionally not including PROGRESS subtype, because it 
        # deterministically means that deal has been closed
        subtypes = [PROPOSAL, IMAGE, WORD, CONTRACT, PDF, ZIP, \
                PDF, NDA, WHITEBOARD, OTHER]
        feature_labels = []
        for t in sorted(types):
            feature_labels.append(t)
        for t in sorted(subtypes):
            feature_labels.append(t)
        return feature_labels
    #enddef


    def __get_doc_features_for_gig(self, gig):
        f_docs = []
        types = [SYSTEM, ATTACHMENT]
        # intentionally not including PROGRESS subtype, because it 
        # deterministically means that deal has been closed
        subtypes = [PROPOSAL, IMAGE, WORD, CONTRACT, PDF, ZIP, \
                PDF, NDA, WHITEBOARD, OTHER]
        type_map = defaultdict(int)
        subtype_map = defaultdict(int)
        for doc in gig[DOCS]:
            type_map[doc[TYPE]] += 1
            subtype_map[doc[SUBTYPE]] += 1
        for t in sorted(types):
            f_docs.append(type_map[t])
        for t in sorted(subtypes):
            f_docs.append(subtype_map[t])
        return f_docs
    #enddef


    def __get_chat_feature_labels(self):
        feature_labels = []
        feature_labels.append('client_to_customer_chat_ratio')
        feature_labels.append('# of unique pm IDs')
        feature_labels.append('inverse exponent of difference of days between 1st & last chat')
        return feature_labels
    #enddef


    def __get_chat_features_for_gig(self, gig):
        f_chats = []
        chats = []
        text = ''
        to_client_count = 0
        pm_ids = []
        for key in gig[CHATS].keys():
            chat = gig[CHATS][key]
            text += chat[TEXT] if TEXT in chat else ''
            if IS_AUTO in chat and chat[IS_AUTO] is True:
                continue
            chat[TIMESTAMP] = datetime.fromtimestamp(float(chat[TIMESTAMP])/1000)
            if chat[TO_CLIENT] is True:
                to_client_count += 1
            if type(chat[PM_ID]) is str:
                pm_ids.append(chat[PM_ID])
            #endif
            chats.append(chat)
        #endfor
        wnl = WordNetLemmatizer()
        #words = [w for w in word_tokenize(text) \
        #        if w not in stopwords.words('english')]
        chats = sorted(chats, key=lambda chat: chat[TIMESTAMP])
        to_client_chat_ratio = 0
        if len(gig[CHATS]) > 0:
            to_client_chat_ratio = float(to_client_count)/len(gig[CHATS])
        f_chats.append(to_client_chat_ratio)
        f_chats.append(len(unique(pm_ids)))
        daydelta = 0
        if len(chats) > 0:
            daydelta = (chats[-1][TIMESTAMP] - chats[0][TIMESTAMP]).days
        f_chats.append(math.exp(-1 * daydelta))
        return f_chats
    #enddef


    def get_feature_labels(self):
        feature_labels = []
        feature_labels.append('Zero Mean Univariate Prices')
        feature_labels.append('Number of platforms')
        feature_labels.append('Number of documents')
        feature_labels.append('Number of key features')
        feature_labels.append('Is Gig Stale?')
        feature_labels += self.__get_doc_feature_labels()
        feature_labels += self.__get_chat_feature_labels()
        return feature_labels
    #enddef


    def __get_features_for_gig(self, gig):
        f_gig = []
        f_gig.append(gig[PRICE])
        f_gig.append(len(gig[PLATFORMS]))
        f_gig.append(len(gig[DOCS]))
        f_gig.append(len(gig[KEY_FEATURES]))
        f_gig.append(1 if gig[STALE] is True else -1 if gig[STALE] is False else 0)
        f_gig += self.__get_doc_features_for_gig(gig)
        f_gig += self.__get_chat_features_for_gig(gig)
        return f_gig
    #enddef


    def get_featured_gigs(self):
        try:
            return self.featured_gigs
        except AttributeError:
            #If the features have not been extracted, we'll do that now.
            pass
        pos_labels = [STARTED, MSHANDOFF, HANDOFF, DONE]
        f_gigs = []
        labels = []
        featured_gigs = {POS:[], NEG:[]}
        for key in self.gigs.keys():
            gig = self.gigs[key]
            f_gigs.append(self.__get_features_for_gig(gig))
            labels.append(1 if gig[STATUS] in pos_labels else -1)
        prices = [f_gig[0] for f_gig in f_gigs]
        prices = self.__make_zero_mean_one_std(prices)
        for i in range(len(prices)):
            f_gigs[i][0] = prices[i]
        for i in range(len(labels)):
            if labels[i] == 1:
                featured_gigs[POS].append(f_gigs[i])
            else:
                featured_gigs[NEG].append(f_gigs[i])
            #endif
        #endfor
        self.featured_gigs = featured_gigs
        return featured_gigs
    #enddef


    def save_relevant_chats(self):
        rel_chats = {}
        with open(self.chatfile) as chats_f:
            json_data = json.load(chats_f)
            self.chats = json_data[MESSAGES]
            for key in self.chats.keys():
                if key in self.gigs:
                    rel_chats[key] = self.chats[key]
            #endfor
        #endwith
        rel_json = {MESSAGES:rel_chats}
        with open('../data/rel_chats.json','w') as rel_f:
            json.dump(rel_json, rel_f, indent=4)
        return None
    #enddef


#endclass
