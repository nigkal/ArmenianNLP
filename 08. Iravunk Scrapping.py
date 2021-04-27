#!/usr/bin/env python
# coding: utf-8



import urllib.request,sys,time
from bs4 import BeautifulSoup
import requests



for i in range(1000, 200780):
    url = 'https://www.iravunk.com/?p=' + str(i) + '&l=am'
    page = requests.get(url)
    text = page.text
    try:
        soup = BeautifulSoup(page.text, 'html.parser')
        links = soup.find_all('p')
        string = []
        for j in range(len(links)):
            string.append(links[j].getText())
        filename = r'user\Iravunk\Iravunk_' + str(i) + '.txt'
        with open(filename,'w', encoding = 'utf-8') as f:
            f.writelines(string)
    except IndexError:
        continue

