#!/usr/bin/env python
# coding: utf-8



import urllib.request,sys,time
from bs4 import BeautifulSoup
import requests
import pandas as pd



links = pd.read_csv('Links.csv')



for i in range(len(links)):
    url = links['Title_URL'][i]
    page = requests.get(url)
    text = page.text
    soup = BeautifulSoup(page.text, 'html.parser')
    link = soup.find_all('p')
    string = []
    for j in range(len(link)):
        t = link[j].getText()
        if t[-1] == '\n': 
            string.append(t)
        else:
            string.append(t + '\n')
    next_link = soup.find_all('a')
    try:
        while next_link[-2].getText() == 'Next':
            url = url.split('?')[0] + next_link[-2]['href']
            page = requests.get(url)
            text = page.text
            soup = BeautifulSoup(page.text, 'html.parser')
            link = soup.find_all('p')
            for j in range(len(link)):
                t = link[j].getText()
                if t[-1] == '\n': 
                    string.append(t)
                else:
                    string.append(t + '\n')
            next_link = soup.find_all('a')
        filename = r'user\Literature_' + links['Title'][i] + '.txt'
        with open(filename,'w', encoding = 'utf-8') as f:
            f.writelines(string)
    except IndexError:
        filename = r'user\Literature_' + links['Title'][i] + '.txt'
        with open(filename,'w', encoding = 'utf-8') as f:
            f.writelines(string)

