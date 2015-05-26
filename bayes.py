# Name: Rene Romo(rgr355) Tessa Haldes(tah210) Camille Warren (cew361)
# Date:5/19/15
# Description: Naive Bayes Implementation using only unigrams, for best performance use pypy
#


from __future__ import division
import math, os, pickle, re, random


#Switch for presence or frequency True if using presence False if frequency
presence = False;
#Specify folder where reviews are stored
folder = "reviews/";
#Threshold for difference to return neutral(If (abs(p(pos) - p(neg)) < threshold) return "neutral"; 
#for this to have any effect uncomment line 133-135
neutral_threshold = 0.1;

class Bayes_Classifier:
   def __init__(self):
      """This method initializes and trains the Naive Bayes Sentiment Classifier.  If a 
      cache of a trained classifier has been stored, it loads this cache.  Otherwise, 
      the system will proceed through training.  After running this method, the classifier 
      is ready to classify input text."""
      if os.path.isfile("positive.dict") and os.path.isfile("positive.count") and os.path.isfile("negative.dict")and os.path.isfile("negative.count")and os.path.isfile("negativedoc.count")and os.path.isfile("positivedoc.count"):
          print "Loading Saved Data"
          #If dictionary files exist then load them now
          self.positive = self.load("positive.dict")
          self.negative = self.load("negative.dict")
          self.posCount = self.load("positive.count")
          self.negCount = self.load("negative.count")
          self.posdocCount = self.load("positivedoc.count")
          self.negdocCount = self.load("negativedoc.count")
          self.prior_pos = self.posdocCount/(self.posdocCount+self.negdocCount)
          self.prior_neg = self.negdocCount/(self.posdocCount+self.negdocCount)
      else:
          print "Training Classifier"
          #else train and create these files
          self.train()

   def smoothing(self,dictionary):
      for x in dictionary :
         dictionary[x] = dictionary[x] + 1
      return dictionary
  
   def train(self, trainingset=[]):   
      """Trains the Naive Bayes Sentiment Classifier."""
      lFileList = trainingset
      positive = {}
      negative = {}
      posCount = 0
      negCount = 0
      negdocCount = 0   
      posdocCount = 0   
      if(trainingset == []):
          print "Training with all files"
          for fFileObj in os.walk(folder):
              lFileList = fFileObj[2]
              break
      else:
          print "Training with trainingset"

      for file in lFileList:
          #for each file load the file as a string and tokenize it 
          path = folder + file
          wordlist = self.loadFile(path)
          wordlist = self.tokenize(wordlist)
          #if we are doing presence instead of frequency then remove any duplicate words in the tokens
          if(presence):
              wordlist = list(set(wordlist))
          #if the file is a 1 star review
          if(file[7] == '1'):        
              #add the number of tokens to our count of negative words and 1 to our count of negative documents
              negdocCount += 1 
              negCount += len(wordlist)
              for word in wordlist:
                  #if the word is already in our negative dictionary increment its value by 1
                  if word in negative:
                     negative[word] += 1 
                  else:
                     #if not in dictionary then add it with a starting value of 1
                     negative[word] = 1
                     if word not in positive:
                         #check if word is in the opposite dictionary already
                         #if not then add it with a starting value of 0
                         positive[word] = 0
          elif(file[7] == '5'):
              #if file is a 5 star review
              #increase positive wordcount and pos doc count
              posdocCount += 1 
              posCount += len(wordlist)
              for word in wordlist:
                  #if the word is already in our positive dictionary increment its value by 1
                  if word in positive:
                     positive[word] += 1 
                  else:
                     #otherwise add it to the dictionary starting value of 1
                     positive[word] = 1
                     if word not in negative:
                         #check if word is in the opposite dictionary already
                         #if not then add it with a starting value of 0
                         negative[word] = 0

      
      
      #set and store values
      self.negative = self.smoothing(negative) 
      self.positive = self.smoothing(positive)
      self.posCount = sum(positive.values())
      self.negCount = sum(negative.values())
      self.posdocCount = posdocCount
      self.negdocCount = negdocCount
      self.prior_pos = self.posdocCount/(self.posdocCount+self.negdocCount)
      self.prior_neg = self.negdocCount/(self.posdocCount+self.negdocCount)
      self.save(positive, "positive.dict")
      self.save(posCount, "positive.count")
      self.save(posdocCount, "positivedoc.count")
      self.save(negative, "negative.dict")
      self.save(negCount, "negative.count")
      self.save(negdocCount, "negativedoc.count")
    
   def classify(self, sText):
      """Given a target string sText, this function returns the most likely document
      class to which the target string belongs (i.e., positive, negative or neutral).
      """
      #get tokens from string
      wordlist = self.tokenize(sText)
      #initialize probability with probability of each type of document
      posProb =  math.log(self.prior_pos)
      negProb =  math.log(self.prior_neg)

      for word in wordlist:
          if word in self.positive: #if word is in one dictionary it's in both
              ###THIS MIGHT BE WRONG I'M not sure if this is how you calculate probability correctly
              posProb += math.log((self.positive[word] /self.posCount))
              negProb += math.log((self.negative[word] / self.negCount))

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
               lTokens.append(sToken)
               sToken = ""
            if c.strip() != "":
               lTokens.append(str(c.strip()))
               
      if sToken != "":
         lTokens.append(sToken)

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

       ##DOUBLE CHECK THESE Calculations
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
        os.rename("positive.dict","positive.dict.temp")
        os.rename("negative.dict","negative.dict.temp")
        os.rename("negative.count","negative.count.temp")
        os.rename("positive.count","positive.count.temp")
        os.rename("negativedoc.count","negativedoc.count.temp")
        os.rename("positivedoc.count","positivedoc.count.temp")
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
        os.remove("positive.dict")
        os.remove("negative.dict")
        os.remove("positive.count")
        os.remove("negative.count")
        os.remove("positivedoc.count")
        os.remove("negativedoc.count")

        os.rename("positive.dict.temp", "positive.dict")
        os.rename("negative.dict.temp", "negative.dict")
        os.rename("negative.count.temp","negative.count")
        os.rename("positive.count.temp","positive.count")
        os.rename("negativedoc.count.temp","negativedoc.count")
        os.rename("positivedoc.count.temp","positivedoc.count")


# def main():
#     #main function which automatically runs

#     bc= Bayes_Classifier();
  
#     bc.ten_fold()
#     return 0;


# if __name__ == "__main__":
#     main()
