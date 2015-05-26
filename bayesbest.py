# Name: Rene Romo(rgr355) Tessa Haldes(tah210) Camille Warren (cew361)
# Date:5/19/15
# Description: Implementation of Naive Bayes Classifier using unigrams and word stems 
# using the NLTK (works best with pypy)


from __future__ import division
from nltk.stem.lancaster import LancasterStemmer
import math, os, pickle, re, random, nltk


#Switch for presence or frequency True if using presence False if frequency
presence = False;
#Specify folder where reviews are stored
folder = "reviews/";
#Threshold for difference to return neutral(If (abs(p(pos) - p(neg)) < threshold) return "neutral"; (uncomment related lines in the classify function)
neutral_threshold = 0.1;

class Bayes_Classifier:
   def __init__(self):
      """This method initializes and trains the Naive Bayes Sentiment Classifier.  If a 
      cache of a trained classifier has been stored, it loads this cache.  Otherwise, 
      the system will proceed through training.  After running this method, the classifier 
      is ready to classify input text."""
      if os.path.isfile("bestpositive.dict") and os.path.isfile("bestpositive.count") and os.path.isfile("bestnegative.dict")and os.path.isfile("bestnegative.count")and os.path.isfile("bestnegativedoc.count")and os.path.isfile("bestpositivedoc.count"):
          print "Loading Saved Data"
          #If dictionary files exist then load them now
          self.positive = self.load("bestpositive.dict")
          self.negative = self.load("bestnegative.dict")
          # self.positivebigrams = self.load("bestpositivebigrams.dict")
          # self.negativebigrams = self.load("bestnegativebigrams.dict")

          self.posCount = self.load("bestpositive.count")
          self.negCount = self.load("bestnegative.count")
          # self.posbigramsCount = self.load("bestpositivebigrams.count")
          # self.negbigramsCount = self.load("bestnegativebigrams.count")

          self.posstemsCount = self.load("bestpositivestems.count")
          self.negstemsCount = self.load("bestnegativestems.count")
          self.positivestems = self.load("bestpositivestems.dict")
          self.negativestems = self.load("bestnegativestems.dict")

          self.posdocCount = self.load("bestpositivedoc.count")
          self.negdocCount = self.load("bestnegativedoc.count")
          self.prior_pos = self.posdocCount/(self.posdocCount+self.negdocCount)
          self.prior_neg = self.negdocCount/(self.posdocCount+self.negdocCount)
      else:
          print "Training Classifier"
          #else train and create these files
          self.train()

   def smoothing(self,dictionary):
      """Function which does +1 smoothing on a dictionary"""
      for x in dictionary :
           dictionary[x] = dictionary[x] + 1
      return dictionary
   
   def train(self, trainingset = []):   
      """Trains the Naive Bayes Sentiment Classifier."""
      lFileList = trainingset
      positive = {}
      negative = {}
      positivebigrams = {}
      negativebigrams = {}
      positivestems = {}
      negativestems = {}
      posCount = 0
      negCount = 0
      negdocCount = 0   
      posdocCount = 0   
      negbigramsCount = 0
      posbigramsCount = 0
      negstemsCount = 0
      posstemsCount = 0
      if(trainingset == []):
        print "Training with all files"
        for fFileObj in os.walk(folder):
            lFileList = fFileObj[2]
            break
      else:
        print "Training with trainingset"
      for file in lFileList:
          path = folder + file
          wordlist = self.loadFile(path)
          wordlist,bigrams,stems = self.get_token_and_grams(wordlist)
          if(presence):
              wordlist = list(set(wordlist))
          if(file[7] == '1'):       
              negdocCount += 1  
              negCount += len(wordlist)
              for word in wordlist:
                  if word in negative:
                     negative[word] += 1 
                  else:
                     #start at 2 instead of 1 (+1 smoothing)
                     negative[word] = 1
                     if word not in positive:
                         #start at 1 instead of 0 +1 smoothing
                         positive[word] = 0
          elif(file[7] == '5'):
              posdocCount += 1 
              posCount += len(wordlist)
              for word in wordlist:
                  if word in positive:
                     positive[word] += 1 
                  else:
                     #start at 2 instead of 1 (+1 smoothing)
                     positive[word] = 1
                     if word not in negative:
                         #start at 1 instead of 0 +1 smoothing
                         negative[word] = 0
      # Bigram training
      ##################################################################
          # if(file[7] == '1'):       
          #     negbigramsCount += len(bigrams)
          #     for gram in bigrams:
          #         if gram in negativebigrams:
          #            negativebigrams[gram] += 1 
          #         else:
          #            negativebigrams[gram] = 1
          #            if gram not in positivebigrams:
          #                positivebigrams[gram] = 0
          # elif(file[7] == '5'):
          #     posbigramsCount += len(bigrams)
          #     for gram in bigrams:
          #         if gram in positivebigrams:
          #            positivebigrams[gram] += 1 
          #         else:
          #            positivebigrams[gram] = 1
          #            if gram not in negativebigrams:
          #                negativebigrams[gram] = 0
      #Stems training
      ####################################################################
          if(file[7] == '1'):       
              negstemsCount += len(stems)
              for stem in stems:
                  if stem in negativestems:
                     negativestems[stem] += 1 
                  else:
                     negativestems[stem] = 1
                     if stem not in positivestems:
                         positivestems[stem] = 0
          elif(file[7] == '5'):
              posstemsCount += len(stems)
              for stem in stems:
                  if stem in positivestems:
                     positivestems[stem] += 1 
                  else:
                     positivestems[stem] = 1
                     if stem not in negativestems:
                         negativestems[stem] = 0








      positive=self.smoothing(positive)
      negative=self.smoothing(negative) 
      posCount=sum(positive.values())
      negCount=sum(negative.values())




      self.negative = negative
      self.positive = positive
      self.posCount = posCount
      self.negCount = negCount

      
      # positivebigrams=self.smoothing(positivebigrams)
      # negativebigrams=self.smoothing(negativebigrams) 
      # self.negativebigrams = negativebigrams
      # self.positivebigrams = positivebigrams


      # posbigramsCount=sum(positivebigrams.values())
      # negbigramsCount=sum(negativebigrams.values())
      # self.posbigramsCount = posbigramsCount
      # self.negbigramsCount = negbigramsCount

      positivestems=self.smoothing(positivestems)
      negativestems=self.smoothing(negativestems)
      self.negativestems = negativestems
      self.positivestems = positivestems
      self.posstemsCount = sum(positivestems.values())
      self.negstemsCount = sum(negativestems.values())



      self.posdocCount = posdocCount
      self.negdocCount = negdocCount
      self.prior_pos = self.posdocCount/(self.posdocCount+self.negdocCount)
      self.prior_neg = self.negdocCount/(self.posdocCount+self.negdocCount)
      self.save(positive, "bestpositive.dict")
      self.save(posCount, "bestpositive.count")
      self.save(negative, "bestnegative.dict")
      self.save(negCount, "bestnegative.count") 

      # self.save(positivebigrams, "bestpositivebigrams.dict")
      # self.save(posbigramsCount, "bestpositivebigrams.count")
      # self.save(negativebigrams, "bestnegativebigrams.dict")
      # self.save(negbigramsCount, "bestnegativebigrams.count") 

      self.save(positivestems, "bestpositivestems.dict")
      self.save(posstemsCount, "bestpositivestems.count")
      self.save(negativestems, "bestnegativestems.dict")
      self.save(negstemsCount, "bestnegativestems.count") 


      self.save(negdocCount, "bestnegativedoc.count")
      self.save(posdocCount, "bestpositivedoc.count")
  
   def ngrams(self, input, n):
      """Given a list of tokens and integer n generates and returns n-grams"""
     output = []
     for i in range(len(input)-n+1):
      g = ' '.join(input[i:i+n])
      output.append(g)
     return output

   def get_token_and_grams(self, sText):
      """return tokens and additional ngrams and stems"""
      tokens = self.tokenize(sText)
      wordlist= tokens
      #bigrams
      bigrams =self.ngrams(tokens,2)

      #stems 
      stems = [] 
      st = LancasterStemmer()
      for word in wordlist:
        stems.append(st.stem(word))

      # wordlist.extend(self.ngrams(tokens,3))
      #trigrams
      #wordlist.extend(self.find_ngrams(wordlist,3))
      return wordlist, bigrams, stems

   def classify(self, sText):
      """Given a target string sText, this function returns the most likely document
      class to which the target string belongs (i.e., positive, negative or neutral).
      """
      #get tokens from string
      wordlist,bigrams,stems = self.get_token_and_grams(sText)
      #initialize probability with probability of each type of document
      posProb =  math.log(self.prior_pos)
      negProb =  math.log(self.prior_neg)

      for word in wordlist:
          if word in self.positive: #if word is in one dictionary it's in both
              posProb += math.log((self.positive[word] /self.posCount))
              negProb += math.log((self.negative[word] / self.negCount))

      # for gram in bigrams:
      #        if gram in self.positivebigrams: #if word is in one dictionary it's in both
      #          posProb += math.log((self.positivebigrams[gram] /self.posbigramsCount))
      #          negProb += math.log((self.negativebigrams[gram] / self.negbigramsCount))


      for stem in stems:
          if stem in self.positivestems: #if word is in one dictionary it's in both
              posProb += math.log((self.positivestems[stem] /self.posstemsCount))
              negProb += math.log((self.negativestems[stem] / self.negstemsCount))
      #neutral classification removing for tenfold crossvalidation
      #diff = posProb-negProb
      #if math.fabs(diff) <= neutral_threshold: 
      #   return "neutral"
      #el

      if posProb > negProb: 
          #if it's more likely that the string is positive return "positive"
         return "positive"
      else: 
         #otherwise return "negative"
         return "negative"

   def loadFile(self, sFilename):
      """Given a file name, return the contents of the file as a string."""

      f = open(sFilename, "r")
      sTxt = f.read()
      f.close()
      return sTxt
   
   def save(self, dObj, sFilename):
      """Given an object and a file name, write the object to the file using pickle."""

      f = open(sFilename, "w")
      p = pickle.Pickler(f)
      p.dump(dObj)
      f.close()
   
   def load(self, sFilename):
      """Given a file name, load and return the object stored in the file."""

      f = open(sFilename, "r")
      u = pickle.Unpickler(f)
      dObj = u.load()
      f.close()
      return dObj

   def tokenize(self, sText): 
      """Given a string of text sText, returns a list of the individual tokens that 
      occur in that string (in order)."""

      lTokens = []
      sToken = ""
      for c in sText:
         if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\"" or c == "_" or c == "-":
            sToken += c
         else:
            if sToken != "":
               lTokens.append(sToken.lower())
               sToken = ""
            if c.strip() != "":
               lTokens.append(str(c.strip()))
               
      if sToken != "":
         lTokens.append(sToken.lower())
      return lTokens

   def evaluate(self,testset):
       """Evaluate the Effectiveness of our Classifier"""
       true_positives = 0
       true_negatives = 0
       false_positives = 0
       false_negatives = 0
       #go through files in the provided test set
       for file in testset:
           path = folder + file
           string = self.loadFile(path)
           result = self.classify(string)

           if (result == "positive"):
               if (file[7] == "5"):
                   #if we classified as positive and the file was positive increment true positives
                    true_positives += 1
               elif(file[7] == "1"):
                   #if we classified as positive and the file was negative increment false positives
                    false_positives += 1 
           elif(result =="negative"):
               if (file[7] == "1"):
                   #if we classified as negative and the file was negative increment true negatives
                    true_negatives += 1
               elif(file[7] == "5"):
                   #if we classified as negative and the file was negative increment false negatives
                    false_negatives += 1       
       
       #calculate precision, recall and fmeasure
       pos_precision = true_positives/(true_positives+ false_positives)
       neg_precision = true_negatives/(true_negatives+ false_negatives)

       pos_recall = true_positives/(true_positives + false_negatives)
       neg_recall = true_negatives/(true_negatives + false_positives)

       pos_fmeasure = (2*pos_precision*pos_recall)/(pos_precision+pos_recall)
       neg_fmeasure = (2*neg_precision*neg_recall)/(neg_precision+neg_recall)

       return [pos_recall,pos_precision, pos_fmeasure, neg_recall, neg_precision, neg_fmeasure]
   
   def ten_fold(self):
        """Perform Ten Fold crossvalidation"""
        pos_recalls=[]
        pos_precisions=[]
        pos_fmeasures = []
        neg_recalls=[]
        neg_precisions=[]
        neg_fmeasures = []
        ##if dictionaries already exist rename them 
        os.rename("bestpositive.dict","bestpositive.dict.temp")
        os.rename("bestnegative.dict","bestnegative.dict.temp")
        os.rename("bestnegative.count","bestnegative.count.temp")
        os.rename("bestpositive.count","bestpositive.count.temp")
        os.rename("bestnegativedoc.count","bestnegativedoc.count.temp")
        os.rename("bestpositivedoc.count","bestpositivedoc.count.temp")
        for fFileObj in os.walk(folder):
            FileList = fFileObj[2]
            break 
        #randomize file list
        random.seed(1)
        random.shuffle(FileList)
        #fold size has to be a tenth of our data set 
        #use floor division to get integer
        fold_size = len(FileList)//10;
        for i in range (10):
            print "Fold: %d "  %i 
            #create a training set and a test set
            testset = FileList[i*fold_size:][:fold_size]
            trainingset = FileList[:i*fold_size] + FileList[(i+1)*fold_size:]
            # print len(trainingset)
            # print len(testset)
            print "Training"
            self.train(trainingset)
            print "Testing"
            results = self.evaluate(testset)
            #store recalls, precisions and fmeasures from this run
            pos_recalls.append(results[0])
            pos_precisions.append(results[1])
            pos_fmeasures.append(results[2])
            neg_recalls.append(results[3])
            neg_precisions.append(results[4])
            neg_fmeasures.append(results[5])

        # #print positive average recall,precision and Fmeasure
        print "Positive:"
        # print "  Recall"
        # print sum(pos_recalls)/len(pos_recalls)
        # print "  Precision"
        # print sum(pos_precisions)/len(pos_precisions)
        print "  F-Measure"
        print sum(pos_fmeasures)/len(pos_fmeasures)

        # #print negative average recall,precision and Fmeasure
        print "Negative:"
        # print "  Recall"
        # print sum(neg_recalls)/len(neg_recalls)
        # print "  Precision"
        # print sum(neg_precisions)/len(neg_precisions)
        print "  F-Measure"
        print sum(neg_fmeasures)/len(neg_fmeasures)


        print "avg_recall"
        r = ((sum(pos_recalls)/len(pos_recalls))+(sum(neg_recalls)/len(neg_recalls)))/2
        print r

        print "avg_precision"
        p =((sum(pos_precisions)/len(pos_precisions))+(sum(neg_precisions)/len(neg_precisions)))/2
        print p
        print "avg_fmeasure"
        print ((sum(pos_fmeasures)/len(pos_fmeasures)) + (sum(neg_fmeasures)/len(neg_fmeasures)))/2
        # print "avg_fmeasure2 "
        # print (2*p*r)/(p+r)
        #Restore old dictionaries
        os.remove("bestpositive.dict")
        os.remove("bestnegative.dict")
        os.remove("bestpositive.count")
        os.remove("bestnegative.count")
        os.remove("bestpositivedoc.count")
        os.remove("bestnegativedoc.count")

        os.rename("bestpositive.dict.temp", "bestpositive.dict")
        os.rename("bestnegative.dict.temp", "bestnegative.dict")
        os.rename("bestnegative.count.temp","bestnegative.count")
        os.rename("bestpositive.count.temp","bestpositive.count")
        os.rename("bestnegativedoc.count.temp","bestnegativedoc.count")
        os.rename("bestpositivedoc.count.temp","bestpositivedoc.count")

# def main():
#     #main function which automatically ver
#     bc= Bayes_Classifier();
#     # bc.ten_fold()
#     return 0;


# if __name__ == "__main__":
#     main()
