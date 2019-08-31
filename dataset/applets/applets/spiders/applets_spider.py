import scrapy
import codecs
import json
from time import sleep
import os
from bs4 import BeautifulSoup


class AppletsSpider(scrapy.Spider):
    name = "applets"

    def start_requests(self):
        with codecs.open("train.urls",'r','utf-8')as fr:
            urls=fr.read().split("\n")
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        page = response.url.split("/")[-1]
        filename = 'crawled/%s.json' % page
        if os.path.isfile(filename):
            self.log('>>>> %s' % filename)
            # sleep(5)
            return
        html = response.body
        soup=BeautifulSoup(html,'html5lib')
        name=soup.find_all("h1","applet-name")
        despt=soup.find_all("p","applet-description")
        permissions=soup.find_all("li","permission-meta")
        meta=soup.find_all("div","installs")
        
        dd = dict()
        dd["url"] = response.url
        # dd['new_url']=new_url
        if name:
            dd['name']=name[0].text
        if despt:
            dd['descrptions']=despt[0].text
        dd["channel"]=[]
        dd["action"]=[]
        if permissions:
            for i in range(len(permissions)):
                dd["channel"].append(permissions[i].h5.text)
                dd["action"].append(permissions[i].span.text)
        if meta:
            dd["count"]=meta[0].span.text.strip()
      
        filename = 'crawled/%s.json' % page
        # if os.path.isfile(filename):
        #     index=0
        #     newfilename = os.path.isfile(filename) +str(index)
        #     while os.path.isfile(newfilename):
        #         index+=1
        #         newfilename = os.path.isfile(filename) +str(index)
        #     filename=newfilename
        #     print("new_filename:", newfilename)
                
        # print(filename)

        with open(filename, 'w') as fw:
            string=json.dumps(dd,indent=4)
            fw.write(string) 
            fw.flush()
            fw.close()
        sleep(0.5)
        # self.log('Saved file %s' % filename)