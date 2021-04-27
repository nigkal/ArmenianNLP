#!/usr/bin/env python
# coding: utf-8


import urllib.request,sys,time
from bs4 import BeautifulSoup
import requests



for i in range(50000, 292095):
    url = 'https://www.panarmenian.net/arm/news/' + str(i)
    page = requests.get(url)
    text = page.text
    try:
        soup = BeautifulSoup(page.text, 'html.parser')
        links = soup.find_all('p')
        if 'PanARMENIAN.Net' in links[0].getText():
            string = []
            string.append(links[0].getText().split('PanARMENIAN.Net - ')[1])
            for j in range(1, len(links) - 1):
                string.append(links[j].getText())
            filename = r'user\PanARMENIAN_' + str(i) + '.txt'
            with open(filename,'w', encoding = 'utf-8') as f:
                f.writelines(string)
    except IndexError:
        continue

