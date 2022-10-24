
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import re
from random import seed
from random import random
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
nltk.download('punkt')
import collections
from nltk.tokenize.treebank import TreebankWordDetokenizer
import os
from sklearn.model_selection import train_test_split

class Prep:

    def __init__(self, data_abs, data_ent, data_re):

        self.data_raw = pd.read_csv(data_abs, sep ='\t',   names = ["PMID", "title", "abstract"],keep_default_na=False)
        self.data_entity = pd.read_csv(data_ent, sep = '\t', names = ["PMID", "Entity number", "type of entity","start char offset", "end offset", "Entity"], keep_default_na=False)
        # self.data_relation = pd.read_csv(data_re, sep = '\t', names = ["PMID", "relation","yOrN", "tye", "Entity1", "Entity2"], keep_default_na=False)
        self.data_relation = pd.read_csv(data_re, sep = '\t', names = ["PMID", "relation", "Entity1", "Entity2"], keep_default_na=False)
        self.data_raw["complete"] = self.data_raw["title"] +" "+ self.data_raw["abstract"]
        self.abs_text = pd.DataFrame(columns = ["PMID", "Tagged abs"])
        self.data_relation['Entity1'] = self.data_relation['Entity1'].apply(lambda x: x.replace('Arg1:', ''))
        self.data_relation['Entity2'] = self.data_relation['Entity2'].apply(lambda x: x.replace('Arg2:', ''))
        self.relations = pd.DataFrame(columns = [ "PMID", "relation", "Entity1","Entity2","strEnt1", "endEnt1", "strEnt2","endEnt2","txt"])
        print("paramerts intialized")

    
    def SentFinder(self,txt, entity):
        
        for index, sentence in enumerate(sent_tokenize(txt)):

            if (entity in sentence):
                return (index)




    def spaceAdder(self, text, entity1, entity2):

        if((entity1 not in entity2) and (entity2 not in entity1) ):
            text = text.replace(entity1, " " + entity1+ " ")
            text = text.replace(entity2, " " + entity2+ " ")




        return (text)

    def spacer(self, text, strEnt1, endEnt1, strEnt2, endEnt2, t1, t2):
        temp = text
     
        Entity1 = text[int(strEnt1):int(endEnt1)]
        Entity2 =  text[int(strEnt2):int(endEnt2)]
      
        
        if((strEnt1 < strEnt2) and (endEnt1 < endEnt2) ):
                
                text = (text)[:int(strEnt1)] + " " + Entity1+ " " + (text)[int(endEnt1):]
                txt1 = (text)[:int(strEnt1)] +  t1 + (text)[int(endEnt1):]

                text = (text)[:int(strEnt2)+2] + " "+  Entity2 + " "+ (text)[int(endEnt2)+2:]              
                txt2 = (text)[:int(strEnt2)+2] +  t2 + (text)[int(endEnt2)+2:]
                
            
        elif((strEnt1 > strEnt2) and (endEnt1 > endEnt2) ):   
                
                text = (text)[:int(strEnt2)] + " "+  Entity2 +" "+ (text)[int(endEnt2):]
                txt2 = (text)[:int(strEnt2)] +  t2 + (text)[int(endEnt2)]

                text = (text)[:int(strEnt1)+2 ] + " " + Entity1+ " " + (text)[int(endEnt1)+2:]
                txt1 = (text)[:int(strEnt1)+2] +  t1 + (text)[int(endEnt1)+2:]

        #entity2 is a substring 0f entity 1
        elif ((strEnt1 <= strEnt2) and (endEnt1 >= endEnt2) ):

                
            text = (text)[:int(strEnt1) ] + " " + Entity1+ " " + (text)[int(endEnt1):]
            txt1 = (text)[:int(strEnt1)] +  t1 + (text)[int(endEnt1):] 

            text = (text)[:int(strEnt2)+1] + " " +  Entity2 + " "+ (text)[int(endEnt2)+2:]
            txt2 = (text)[:int(strEnt2)+1] +  t2 + (text)[int(endEnt2)+2:]

            Entity1 = temp[int(strEnt1):int(strEnt2)] + " "  + temp[int(endEnt2)+1:int(endEnt1)]

        #entity 1 substring of entity 2        
        elif ((strEnt2 <= strEnt1) and (endEnt1 <= endEnt2) ):
 
            text = (text)[:int(strEnt2) ] + " " + Entity2+ " " + (text)[int(endEnt2):]
            txt2 = (text)[:int(strEnt2)] +  t2 + (text)[int(endEnt2)] 
            
            text = (text)[:int(strEnt1)+1] + " " +  Entity1 + " "+ (text)[int(endEnt1)+2:] 
            txt1 = (text)[:int(strEnt1)+1] +  t1 + (text)[int(endEnt1)+2:] 
            Entity2 = temp[int(strEnt2):int(strEnt1)-1] + " "  + temp[int(endEnt1)+1:int(endEnt2)]     
            
 
        return (Entity1, Entity2, txt1, txt2,text)
          


    def text_cleaner(self, text):
        punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~+'''
        for ele in text:
            if ele in punc:
                text = text.replace(ele, "")
        



        text = TreebankWordDetokenizer().detokenize(word_tokenize(text))
        return(text)

 



    def prerocessing(self):
        ll=0
        for i in range(len(self.data_raw)):

        # for i in range(4):    

            #extraction all entities and relations of a specific PMID and iterate through it
            pmid = self.data_raw.loc[i,"PMID"]
            i_entities = (self.data_entity[self.data_entity["PMID"] == pmid]).copy()
            i_relations =(self.data_relation[self.data_relation["PMID"] ==pmid]).copy()
            text = self.data_raw.loc[i,"complete"]
            tempText = text
            i_relations = i_relations.reset_index(drop = True)

            for k in range(len(i_relations)):
                text = self.data_raw.loc[i,"complete"]
                i_relations.loc[k,"strEnt1"]= int(i_entities.loc[(i_entities["Entity number"]== str(i_relations.loc[k,"Entity1"])), ["start char offset"]].values[0][0])
                i_relations.loc[k,"endEnt1"]= int(i_entities.loc[(i_entities["Entity number"]== str(i_relations.loc[k,"Entity1"])), ["end offset"]].values[0][0])
                i_relations.loc[k,"strEnt2"]= int(i_entities.loc[(i_entities["Entity number"]== str(i_relations.loc[k,"Entity2"])), ["start char offset"]].values[0][0])
                i_relations.loc[k,"endEnt2"]= int(i_entities.loc[(i_entities["Entity number"]== str(i_relations.loc[k,"Entity2"])), ["end offset"]].values[0][0])


                t1 = "<" + i_relations.loc[k, "Entity1"] + ">"
                t2 = "</" + i_relations.loc[k,"Entity2"] +">"                                    
             

                strEnt1 = i_relations.loc[k,"strEnt1"]
                endEnt1 = i_relations.loc[k,"endEnt1"]
                strEnt2 = i_relations.loc[k,"strEnt2"]
                endEnt2 = i_relations.loc[k,"endEnt2"]
                    
                e1, e2, txt1, txt2, text = self.spacer(text, strEnt1, endEnt1, strEnt2, endEnt2, t1, t2)
                strSent = self.SentFinder( txt1 , t1)
                endSent = self.SentFinder(txt2 , t2)
         
                if(strSent != endSent):
                    ll=ll+1
                    continue
                                    
                tempText = nltk.tokenize.sent_tokenize(text)
                  
                i_relations.loc[k,"Entity1"]=  e1
                i_relations.loc[k,"Entity2"] = e2
                i_relations.loc[k,"txt"]= tempText[strSent]
                i_relations.at[k,"sentNumber"] = strSent

              

                i_relations.loc[k,"txt"] = self.text_cleaner(i_relations.loc[k,"txt"])
                i_relations.loc[k,"Entity1"] = self.text_cleaner(i_relations.loc[k,"Entity1"])
                i_relations.loc[k,"Entity2"] = self.text_cleaner(i_relations.loc[k,"Entity2"])

                    
                ent1_position = i_relations.loc[k,"txt"].find(i_relations.loc[k,"Entity1"])
                ent2_position = i_relations.loc[k,"txt"].find(i_relations.loc[k,"Entity2"])

                if (ent1_position > ent2_position ):
                    i_relations.at[k,"Entity1"], i_relations.at[k,"Entity2"] = i_relations.loc[k,"Entity2"] , i_relations.loc[k,"Entity1"]


            self.relations = self.relations.append(i_relations)
                
        self.relations = self.relations.reset_index(drop = True)
        #if there are still records with txt longer than 512 delete them :)
        mask = (self.relations['txt'].str.len() < 512)
        self.relations = self.relations.loc[mask]
        print("ll value is: ", ll)
        print("Len relations: ", len(self.relations))
        print("Preprocessing is done!")
        return(self.relations)

 



class LoadData:
    def __init__(self, out_file, normal_file, epo_file, seo_file, relations):

        self.out_file = out_file
        self.normal_file = normal_file
        self.epo_file = epo_file
        self.seo_file = seo_file
        self.relations = relations

    def fixer(self):    

        id_checker = set()
        for i in range(len(self.relations)):
            sent_num = self.relations.iloc[i]["sentNumber"]
            pmid = self.relations.iloc[i]["PMID"]
            id_rec = (str(pmid)+ str(sent_num))
            
            if id_rec not in id_checker:
                id_checker.add(id_rec)
                
                
            

                a = self.relations[(self.relations["PMID"]== pmid) &(self.relations["sentNumber"]== sent_num)].index
                
                max_len = len(self.relations.iloc[int(a[0])]["txt"])
                new_idx = 0
                for idx in range(len(a)):
                    new_len = len(self.relations.iloc[int(idx)]["txt"])
                    if (new_len > max_len):
                        new_idx = idx

                text = self.relations.iloc[int(a[int(new_idx)])]["txt"]

                entities=set()

                for num in a:
                    entities.add(self.relations.iloc[int(num)]["Entity1"])
                    entities.add(self.relations.iloc[int(num)]["Entity2"])

                changed =[]
                for x in entities:
                    entity_diff = (entities.difference(*[[x]]))
                    for value in entity_diff:
                        space_x = " "+ x +" "
                        space_value = " "+ value+ " "
                        if(x in value) :
                            item =dict()

                            if (space_x not in text ):
                                text = text.replace(x, space_x)
                                item["pre"]= value
                                item["new"] = value.replace(x, space_x)
                                changed.append(item)
                        else:                                      
                                if(space_value not in text):
                                    text= text.replace(value,space_value)
                                if(space_x not in text):
                                    text = text.replace(x, space_x)
                                    
                                    
                            

                text = TreebankWordDetokenizer().detokenize(word_tokenize(text))
                for idx in a:
                    self.relations.at[int(idx), "txt"]= text
                    ent1 = self.relations.iloc[int(idx)]["Entity1"]
                    ent2 = self.relations.iloc[int(idx)]["Entity2"]
                    for item in changed:
                        if (item["pre"] == ent1):
                            self.relations.at[int(idx), "Entity1"]= (item["new"]).strip()
                        if (item["pre"] == ent2):
                            self.relations.at[int(idx), "Entity2"]= (item["new"]).strip()
        
                            
                        


    def load(self):
        
        
        with open ( self.out_file, 'w') as f2, open(self.normal_file, 'w') as f3, open(self.epo_file, 'w') as f4, open(self.seo_file, 'w') as f5:
           
            unique_texts = self.relations['txt'].unique()
        
            for text in unique_texts:
                temp = self.relations[self.relations["txt"]== text].reset_index(drop = True)

                
                new_line = dict()
                relationMentions = []
                sentText = text
                
                for i in range (len(temp)):

                    rel = dict()
                    rel['em1Text'] = temp.iloc[i]["Entity1"]
                    rel['em2Text'] = temp.iloc[i]["Entity2"]
                    rel['label'] = temp.iloc[i]["relation"]
                    
                    #deleting typos!
                    if((rel['em1Text'] in sentText) & (rel['em2Text'] in sentText)):
                        if (rel not in relationMentions):
                            relationMentions.append(rel)

                    
                        
                if(len(relationMentions) > 0):  
                    new_line['sentText'] = sentText
                    new_line['relationMentions'] = relationMentions
                    f2.write(json.dumps(new_line)+'\n')

                if(len(new_line)> 0 and len(relationMentions) > 0 ):
                    self.is_seo_and_normal(new_line, relationMentions, f3,f4, f5)
                    

        print("Load data is done!")





    def is_seo_and_normal(self, new_line, relationMentions, f3,f4, f5 ):
    
    #     first entities be common
        firstEntities = [x['em1Text'] for x in relationMentions ]
        secondEntities = [x['em2Text'] for x in relationMentions ]
        labels = [x['label'] for x in relationMentions ]
        
        intrsct_firstEntities = [item for item, count in collections.Counter(firstEntities).items() if count > 1]
        index = "em1Text"
        res1 = self.intersect_writer(new_line['sentText'], intrsct_firstEntities, relationMentions,index ,f5)
   

    #     second entities be common   
        intrsct_secEntities = [item for item, count in collections.Counter(secondEntities).items() if count > 1]
        index = "em2Text"
        res2 = self.intersect_writer(new_line['sentText'], intrsct_secEntities, relationMentions, index ,f5)
 

        
    #     if it is normal not have common first entities or second inteties
        if(not(res1 or res2)):
            f3.write(json.dumps(new_line)+ '\n')

                
        new_eop_line = dict()
        rel_multi =[]
    #     fisrt and second entities together, multilabel    
        if(res1 and res2):
            
            for i in range(len(firstEntities)):
                
                for j in range(i,(len(firstEntities)-1)):
                    
                    if((firstEntities[i] == firstEntities[j+1])& ( secondEntities[i] == secondEntities[j+1] ) & (labels[i] != labels[j+1])):
                        rel_multi = [x for x in relationMentions if ((x['em1Text']== firstEntities[i]) & (x['em2Text'] == secondEntities[i]))]                   
            if(len(rel_multi) > 0):            
                new_eop_line['sentText'] = new_line['sentText']
                new_eop_line ['relationMentions'] = relationMentions
                f4.write(json.dumps(new_eop_line)+ '\n') 





    def intersect_writer(self,txt, intrsct_Entities, relationMentions, index,f5):
    
        new_seo_line = dict()
        if (len(intrsct_Entities) > 0  and len(relationMentions) > 0):
            # for item in intrsct_Entities:
            #     rel_seo = [x for x in relationMentions if x[index]== item]
            # for item in relationMentions:
            rel_seo = [x for x in relationMentions ]
            new_seo_line['sentText'] = txt
            new_seo_line ['relationMentions'] = rel_seo
            f5.write(json.dumps(new_seo_line)+ '\n')
            return(True)
        else:
            return(False)
    


    def random_creator():
        rand_num = random()
        if(rand_num > 0.8):
            return(True)
        elif(rand_num <= 0.8):
            return(False)
            

if __name__ == '__main__':
  
    # output = 'train.json'
    # normal_file = 'train_normal.json'
    # epo_file = 'train_epo.json'
    # seo_file = 'train_seo.json'
    
    # train = Prep("chemprot_training_abstracts.tsv","chemprot_training_entities.tsv", "chemprot_training_gold_standard.tsv")
    # train_Relations = train.prerocessing()
    # train_Relations.to_csv("relationTrain.csv", index=False, header=True)
    # train_Relations= pd.read_csv("relationTrain.csv", keep_default_na=False)  
     
    # trainLoad = LoadData(output, normal_file, epo_file, seo_file, train_Relations)
    # trainLoad.fixer()
    # trainLoad.load()


#########################################################################################
    # output = 'dev.json'
    # normal_file = 'dev_normal.json'
    # epo_file = 'dev_epo.json'
    # seo_file = 'dev_seo.json'
    
    # train = Prep("chemprot_development_abstracts.tsv","chemprot_development_entities.tsv", "chemprot_development_gold_standard.tsv")
    # train_Relations = train.prerocessing()
    # train_Relations.to_csv("relationDev.csv", index=False, header=True)
    # train_Relations= pd.read_csv("relationDev.csv", keep_default_na=False)

    # trainLoad = LoadData(output, normal_file, epo_file, seo_file, train_Relations)
    # trainLoad.fixer()
    # trainLoad.load()
    ##########################################################################

    output = 'test.json'
    normal_file = 'test_normal.json'
    epo_file = 'test_epo.json'
    seo_file = 'test_seo.json'

    # train = Prep("chemprot_test_abstracts_gs.tsv","chemprot_test_entities_gs.tsv", "chemprot_test_gold_standard.tsv")
    # train_Relations = train.prerocessing()
    # train_Relations.to_csv("relationTest.csv", index=False, header=True)

    train_Relations= pd.read_csv("relationTest.csv", keep_default_na=False)
    trainLoad = LoadData(output, normal_file, epo_file, seo_file, train_Relations)
    trainLoad.fixer()
    trainLoad.load()

###################################################################
    # all_data = [output, normal_file, epo_file, seo_file]
    # train_files = ['train.json', 'train_normal.json', 'train_epo.json', 'train_seo.json' ]
    # test_files = ['test.json', 'test_normal.json', 'test_epo.json', 'test_seo.json']

    
    # # split 20% of the data for test
    # for i in range (len(all_data)):
    #     with open(all_data[i], 'r') as main_file, open(train_files[i], 'w') as train_f, open(test_files[i], 'w') as test_f:
    #         file_lines = main_file.readlines()

    #         if (len(file_lines) > 0):
                
    #             train , test = train_test_split(file_lines, test_size = 0.2)
    #             for j in train:
    #                 train_f.write(j)

    #             for k in test:
    #                 test_f.write(k)
    #     os.remove("{}".format(all_data[i]))
####################################################################
    # generate the validation data
    # devlp = Prep("drugprot_development_abstracs.tsv", "drugprot_development_entities.tsv", "drugprot_development_relations.tsv")
    # devlp_relations = devlp.prerocessing()
    # devLoad = LoadData(output, normal_file, epo_file, seo_file, devlp_relations)
    # devLoad.load()





           
                



    
