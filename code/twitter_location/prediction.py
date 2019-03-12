#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import re
import unidecode
import sys
import os
import string
import spacy
import json
import warnings
import urllib

def queryOnline(query):    
    url = "https://api.duckduckgo.com/?q={}&format=json".format(query)
    try:
        response = urllib.urlopen(url)
    except:
        return None
    try:
        data = json.loads(response.read())
    except:
        return None
    if(len(data["RelatedTopics"]) != 0):
        if("Text" in data["RelatedTopics"][0] and len(data["RelatedTopics"][0]["Text"]) !=0):
            return data["RelatedTopics"][0]["Text"].split("by")[0]


nlp = spacy.load('xx_ent_wiki_sm')
def locationsInQuery(query,nlp=nlp):
    locations = []
    result = queryOnline(query)
    if(result):
        doc = nlp(result)
        for ent in doc.ents:
            if(ent.label_ == u"LOC"):
                locations.append(ent.text.lower())
        doc = nlp(" ".join(result.split(" ")[0:2]))
        for ent in doc.ents:
            if(ent.label_ == u"LOC"):
                locations.append(ent.text.lower())
        if(len(locations)>0):
            return locations
warnings.filterwarnings("ignore")

# Tokenizer for location strings
def loc_tokenizer(str):
    tokens = []
    tbl = string.maketrans('?!.\'', '    ')
    str = str.translate(tbl)
    for token in re.split('\)|;|-| |,|/|\(',str):
        if(len(token)>0):
            tokens.append(token)
    return tokens

# Keyword generator to geneterate keywords from location string
def keywordGenerator(maxTokenInWord,str):
    words = []
    tokens = loc_tokenizer(str)
    if(len(tokens) == 1):
        return tokens
    for i in range(1,maxTokenInWord+1):
        for j in range(len(tokens)-i+1):
            words.append(" ".join(tokens[j:j+i]))
    return words

# Naive Bayes Probability Calculation Functions  

def checkForCountryInKeywords(word,df_country,df_prediction,eps):
    if(word in df_country):
        code = df_country[word]
        df_prediction[code] = df_prediction[code]/eps
        df_prediction = df_prediction * eps
    return df_prediction

def checkForCityInKeywords(word,df_city,df_prediction,eps):
    if(word in df_city):
        city_populations = df_city[word]
        populations = pd.Series(data = [eps]*len(df_prediction),index = df_prediction.index)
        populations.update(city_populations/float(10**8))
        df_prediction = df_prediction.multiply(populations)
    return df_prediction

def checkForRegionInKeywords(word,df_region,df_prediction,eps):
    if(word in df_region):
        region_populations = df_region[word]
        populations = pd.Series(data = [eps]*len(df_prediction),index = df_prediction.index)
        populations.update(region_populations/float(10**8))
        df_prediction = df_prediction.multiply(populations)
    return df_prediction

def checkForLanguage(lang,df_CClang,df_prediction,eps):
    eps = 0.01
    if(len(lang)>0):
        langs = pd.Series(data = [eps]*len(df_prediction),index = df_prediction.index)
        lang_count = df_CClang.groupby("CC").size()[df_CClang[df_CClang["lang"] == lang]["CC"]]
        lang_count = lang_count[~lang_count.index.duplicated()]
        langs.update(1.0/lang_count)
        df_prediction = df_prediction.multiply(langs)
    return df_prediction
def checkForState(keywords,us_states,df_prediction):
    confidence = 10
    if(df_prediction.min() != df_prediction["us"]):
        confidence = 1000
    for word in keywords:
        if(word in us_states):
            df_prediction["us"] = df_prediction["us"]*confidence
            return df_prediction
    return df_prediction


def searchOnline2(location,df_prediction,df_country,df_city,df_region,eps):
    keywords = keywordGenerator(2,location)
    result =[]
    if(keywords):
        for word in keywords:
            search = locationsInQuery(word)
            if(search):   
                result.extend(search)
    if(result):
        for key in result:
            df_prediction = checkForCountryInKeywords(key,df_country,df_prediction,eps)
            df_prediction = checkForCityInKeywords(key,df_city,df_prediction,eps)
            df_prediction = checkForRegionInKeywords(key,df_region,df_prediction,eps)
    return df_prediction
# Naive Bayes Function
def NaiveBayesLocationLanguageState(user,df_city,df_region,df_country,populationCC,df_CClang,country_codes,us_states,n):
    eps_in = 0.00001
    location = unidecode.unidecode(user["location"]).lower()
    lang = user["lang"].lower()
    keywords = keywordGenerator(n,location)
    df_prediction = pd.Series(data=np.array([1]*len(country_codes)),index=country_codes)
    if(len(keywords) != 0):    
        for word in keywords:
            eps = eps_in *10**-(2*word.count(" "))
            df_prediction = checkForCountryInKeywords(word,df_country,df_prediction,eps)
            df_prediction = checkForCityInKeywords(word,df_city,df_prediction,eps)
            df_prediction = checkForRegionInKeywords(word,df_region,df_prediction,eps)
        checkForState(keywords,us_states,df_prediction)
        if(sum(df_prediction == 1) == len(country_codes)):
            df_prediction = searchOnline2(location,df_prediction,df_country,df_city,df_region,eps)
    if("en" not in lang):
        eps = eps_in
        df_prediction = checkForLanguage(lang,df_CClang,df_prediction,eps)
    if(sum(df_prediction == 1) == len(country_codes)):
        return None
    minValue = df_prediction.min()
    df_prediction[df_prediction == minValue] = 0.0
    df_prediction = df_prediction.multiply(populationCC,fill_value = eps)
    return df_prediction

def clearAmbiguity(keywords,user,nlp=nlp):
    doc = nlp(user["location"])
    locs = []
    for ent in doc.ents:
        if(ent.label_ == u"LOC"):
            locs.append(unidecode.unidecode(ent.text).lower())
    keywords.extend(locs)
    return keywords

# Naive Bayes Function2
def NaiveBayesLocationLanguageState2(user,df_city,df_region,df_country,populationCC,df_CClang,country_codes,us_states,n):
    eps_in = 0.00001
    location = unidecode.unidecode(user["location"]).lower()
    lang = user["lang"].lower()
    keywords = keywordGenerator(n,location)
    df_prediction = pd.Series(data=np.array([1]*len(country_codes)),index=country_codes)
    if(len(keywords) == 0):
        keywords = clearAmbiguity(keywords,user)
        for word in keywords:
            eps = eps_in *10**-(2*word.count(" "))
            df_prediction = checkForCountryInKeywords(word,df_country,df_prediction,eps)
            df_prediction = checkForCityInKeywords(word,df_city,df_prediction,eps)
            df_prediction = checkForRegionInKeywords(word,df_region,df_prediction,eps)
        checkForState(keywords,us_states,df_prediction)
        if(sum(df_prediction == 1) == len(country_codes)):
            df_prediction = searchOnline2(location,df_prediction,df_country,df_city,df_region,eps)
    if("en" not in lang):
        eps = eps_in
        df_prediction = checkForLanguage(lang,df_CClang,df_prediction,eps)
    if(sum(df_prediction == 1) == len(country_codes)):
        return None
    minValue = df_prediction.min()
    df_prediction[df_prediction == minValue] = 0.0
    df_prediction = df_prediction.multiply(populationCC,fill_value = eps)
    return df_prediction

def predict_location(user):
    df_prediction = NaiveBayesLocationLanguageState(user,df_city,df_region,df_country,populationCC,df_CClang,country_codes,us_states,3)
    if(type(df_prediction) != type(None)):
        return df_prediction.index[df_prediction == df_prediction.max()][0]
def predict_location2(user):
    df_prediction = NaiveBayesLocationLanguageState(user,df_city,df_region,df_country,populationCC,df_CClang,country_codes,us_states,3)
    if(type(df_prediction) != type(None)):
        return df_prediction.index[df_prediction == df_prediction.max()][0]

def possible_location_distribution(user):
    df_prediction = NaiveBayesLocationLanguageState(user,df_city,df_region,df_country,populationCC,df_CClang,country_codes,us_states,3)
    if(type(df_prediction) != type(None)) :
        df_possible = df_prediction.nlargest(5)
        df_possible = df_possible/df_possible.sum()
        return df_possible

def predict(location,language = "en"):
    user = {"location": location,"lang": language}
    return predict_location(user)

def predict_top_five(location,language = "en"):
    user = {"location": location,"lang": language}
    return possible_location_distribution(user)

data_dir = "/Users/onurkilicoglu/Desktop/MIKS_Internship/location_prediction/data"
# Reading city,region country and population data 
df = pd.read_csv(os.path.join(data_dir,"worldcities.txt"))

global country_codes
country_codes = df["CC"].unique().tolist()
country_names = {}
language = []
country_population = {}

# Reading language and country population data
with open(os.path.join(data_dir,"country_info.txt"), 'rb') as f:
    f.readline()
    for line in f:
        row = line.translate(None,"\n").lower().split("\t")
        code = row[0]
        country = row[4]
        if(sum(df["country"] == country) == 0):
            country_names[country] = code
            if(code not in country_codes):
                country_codes.append(code)
        population = float(row[7])
        languages = row[15].split(",")
        for lang in languages:
            language.append([code,lang])
            lang = lang.split("-")
            if(len(lang) == 2):
                language.append([code,lang[0]])
            if(len(lang)>1):
                lang[1] = lang[1].upper()
            langCode = "_".join(lang)
            if(os.path.isdir(os.path.join(data_dir,"country-list-master/data",langCode)) and len(langCode)>0):
                with open(os.path.join(data_dir,"country-list-master/data",langCode,"country.json")) as f:
                    mapping = json.load(f)
                    name_in_lang = unidecode.unidecode(mapping[code.upper()]).lower()
                    country_names[name_in_lang] = code
        country_population[code] = population

# Reading names of countries in different languages
for langCode in "ms,es,ja,pt,ar,fr,tr,ko,th,de".split(","):    
    with open(os.path.join(data_dir,"country-list-master/data",langCode,"country.json")) as f:
        mapping = json.load(f)
        for code in mapping:
            if(code in country_codes):
                name_in_lang = unidecode.unidecode(mapping[code]).lower()
                country_names[name_in_lang] = code.lower()
# Reading state abbreviations of states in USA 
global us_states
us_states = []
with open(os.path.join(data_dir,"states.txt")) as f:
    for line in f:
        us_states.append(line.translate(None,"\n").lower())
us_states = set(us_states)

# Organizing language and population data
global df_CClang
df_CClang = pd.DataFrame(data = np.array(language),columns = ["CC","lang"])
global populationCC
populationCC = pd.Series(data = country_population)

# Organizing city, region and country data
regions = df["region_name"][df['region_name'].notna()].unique().tolist()
cities = df["city_name"][df['city_name'].notna()].unique().tolist()
countries = df["country"].unique().tolist()

region_groups = df.groupby("region_name")
city_groups = df.groupby("city_name")
region_data = {}
city_data = {}

for region in regions:
    population = region_groups.get_group(region).groupby("CC")["population"].sum()
    for code in population.index:
        region_data[(region,code)] = population[code]
global df_region
df_region = pd.Series(data = region_data)

for city in cities:
    population = city_groups.get_group(city).groupby("CC")["population"].sum()
    for code in population.index:
        city_data[(city,code)] = population[code]
global df_city
df_city = pd.Series(data = city_data)

global df_country
df_country = pd.Series(index=countries)
for country in country_names:
    df_country[country] = country_names[country]
for country in countries:
    df_country[country] = df[df["country"] == country]["CC"].tolist()[0]