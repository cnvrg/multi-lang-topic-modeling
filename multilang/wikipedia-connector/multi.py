import wikipedia
import pandas as pd
import re
import requests
import urllib.request
import time
from bs4 import BeautifulSoup
import numpy as np
import argparse
from urllib.request import urlopen
from urllib.error import HTTPError
import lxml
import os
import translators as ts

parser = argparse.ArgumentParser(description="""Preprocessor""")
parser.add_argument('-f','--topics', action='store', dest='topics', default="table,chalk,starcraft,predator", required=True, help="""wikipedia topics""")
parser.add_argument('--project_dir', action='store', dest='project_dir',
                        help="""--- For inner use of cnvrg.io ---""")
parser.add_argument('--output_dir', action='store', dest='output_dir',
                        help="""--- For inner use of cnvrg.io ---""")

args = parser.parse_args()
topic = args.topics
cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")
if topic.endswith(".csv"):
    topics = pd.read_csv(topic)
    flag_0 = "dataframe"
else:
    topics = topic.split(",")
    flag_0 = "list"


class WikiPage:
    def __init__(self, page):
        ''' Stores the page name and calls the get_wiki_page() function '''
        if page is not None:
            self.page = page
        else:
            raise ValueError("Please specify a page or check if the page is valid")

    def get_wiki_page(self, page):
        ''' Uses the wikipedia library to get the content of the page specified as variable. Calls the get_clean_text() function and returns the cleaned text '''
        output = []
        flag = 0
        if ".org" in page:
            if page[8:10] in ('zh'):
                encoded_url = urllib.parse.urlencode({'1': page.split("/wiki/")[1]}, encoding='utf8')
                encoded_url = 'https://zh.wikipedia.org/wiki/'+encoded_url[2:]
                soup = BeautifulSoup(urlopen(encoded_url).read(), 'lxml')
                text = ''
                for paragraph in soup.find_all('p'):
                    text = text + paragraph.text
                cleaned_text = self.chinese_text_processing(text)
                if(len(cleaned_text) < 100):
                    flag = 1
            if page[8:10] in ('en'):
                soup = BeautifulSoup(urlopen(page).read(), 'lxml')
                text = ''
                for paragraph in soup.find_all('p'):
                    text += paragraph.text
                cleaned_text = self.get_clean_text(text)
                if(len(cleaned_text) < 100):
                    flag = 1
            if page[8:10] in ('fr','de','it'):
                encoded_url = urllib.parse.urlencode({'1': page.split("/wiki/")[1]}, encoding='utf8')
                lang_code_var = page[8:10]
                encoded_url = 'https://'+lang_code_var+'.wikipedia.org/wiki/'+str(encoded_url[2:])
                soup = BeautifulSoup(urlopen(encoded_url).read(), 'lxml')
                text = ''
                for paragraph in soup.find_all('p'):
                    text = text + paragraph.text
                cleaned_text = self.euro_text_processing(text,lang_code_var)
                if(len(cleaned_text) < 100):
                    flag = 1
            output.append(cleaned_text)
            output.append(str(flag))
            return output

        else:
            cnt = 0
            done = 0
            while done<1:
                wik_url = wikipedia.search(page)[cnt].replace(' ','_')
                wiki = 'https://en.wikipedia.org/wiki/'+str(wik_url)
                soup = BeautifulSoup(urlopen(wiki).read(), 'lxml')
                text = ''
                for paragraph in soup.find_all('p'):
                    text += paragraph.text
                cleaned_text = self.get_clean_text(text)
                if(len(cleaned_text) > 100):
                    done = 1
                else:
                    flag = 1
                    cnt = cnt+1
            output.append(cleaned_text) #type(output) type(cleaned_text) output[1]
            output.append(str(flag))
            return output

    def get_clean_text(self, text):
        ''' Cleans the text content from wiki pge using regex '''
        text = re.sub(r'\[.*?\]+', '', text)
        text = text.replace('\n', ' ')
        text = re.sub(r'[^\x00-\x7F]+',' ', text)
        return text
    def chinese_text_processing(self, text):
        ################# text cleaning #############
        text = re.sub(r'\[.*?\]+', '', text)
        text = re.sub(r'\s*[A-Za-z]+\b',    '', text)
        #processed_text = re.sub(r'[^\u4E00-\u9FA5]+', '', processed_text)
        translated_words = ''
        n = 4500
        lang_code = 'zh-CN'
        ################# text breaking up ###########
        list_words = [text[i:i+n] for i in range(0, len(text), n)]
        ################# text translation and rejoining #################
        for texts in list_words:
            translated = ts.google(
                texts, from_language=lang_code, to_language='en')
            translated_words = translated_words + translated
        return translated_words
    def euro_text_processing(self, text, lang_code_variable):
        ############# text cleaning #############
        text = re.sub(r'\[.*?\]+', '', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        #text3 = re.sub(r'\s*[A-Za-z]+\b',    '', text22)
        #text4 = re.sub(r'[^\u4E00-\u9FA5]+', '', text22)
        cnt_sent = 0
        cnt_stop = 0
        list_words = []
        lang_code = lang_code_variable
        translated_words = ''
        for i in range(len(text)):
            if text[i] == '.' and (text[i-1].isnumeric() == False):
                cnt_sent = cnt_sent+1
            if cnt_sent > 10:
                list_words.append(text[cnt_stop:i])
                cnt_stop = i
                cnt_sent = 0
        for texts in list_words:
            translated = ts.google(
                texts, from_language=lang_code, to_language='en')
            translated_words = translated_words + translated
        return translated_words

try:
    if flag_0 == "dataframe":
        input_list = []
        list2 = []
        disambiguation = 0
        if(topics.shape[0]>0):
            for i in range(topics.shape[0]):
                abc1 = WikiPage(str(topics.iloc[i, 0]))
                abc11 = abc1.get_wiki_page(str(topics.iloc[i, 0]))[0]
                input_list.append(abc11)
                if(abc1.get_wiki_page(str(topics.iloc[i, 0]))[1] == '1'):
                    disambiguation = disambiguation + 1
                list2.append(str(topics[i]))
        else:
            for j in range(len(topics.columns.values)):
                abc2 = WikiPage(str(topics.columns.values[j]))
                abc22 = abc2.get_wiki_page(str(topics.columns.values[j]))[0]
                input_list.append(abc22)
                if(abc2.get_wiki_page(str(topics.columns.values[j]))[1] == '1'):
                    disambiguation = disambiguation + 1
                list2.append(str(topics[i]))
        df1 = pd.DataFrame(list(zip(input_list, list2)), columns =['text', 'title'])
        print('There were ' + str(disambiguation) + ' ambigious values in the input list')
        output_summaries_file = "wiki_output.csv"
        df1.to_csv(cnvrg_workdir+"/{}".format(output_summaries_file), index=False)
    else:
        input_list = []
        list2 = []
        disambiguation = 0
        for i in range(len(topics)):
            abc1 = WikiPage(str(topics[i]))
            abc11 = abc1.get_wiki_page(str(topics[i]))[0]
            input_list.append(abc11)
            if(abc1.get_wiki_page(str(topics[i]))[1] == '1'):
                disambiguation = disambiguation + 1
            list2.append(str(topics[i]))
        #lst2 = ['x'] * len(input_list)
        df1 = pd.DataFrame(list(zip(input_list, list2)), columns =['text', 'title'])
        print('There were ' + str(disambiguation) + ' ambigious values in the input list')
        output_summaries_file = "wiki_output.csv"
        df1.to_csv(cnvrg_workdir+"/{}".format(output_summaries_file), index=False)

except (wikipedia.exceptions.PageError, wikipedia.exceptions.WikipediaException,urllib.error.HTTPError):
    print('Does not match any page. Try another ID next time')