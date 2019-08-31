import scrapy
from bs4 import BeautifulSoup


class QuotesSpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):
        with codecs.open("train.urls",'r','utf-8')as fr:
            urls=fr.read().split("\n")
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        page = response.url.split("/")[-2]
        html = response.body
        soup=BeautifulSoup(html,'html5lib')
        name=soup.find_all("h1","applet-name")
        despt=soup.find_all("p","applet-description")
        permissions=soup.find_all("li","permission-meta")
        meta=soup.find_all("div","installs")
        
        dd = dict()
        dd["url"] = url
        dd['new_url']=new_url
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
      
        filename = 'quotes-%s.json' % page
        print(filename)
        with open(filename, 'w') as f:
            string=json.dumps(dd,indent=4)
            fw.write(string) 
            fw.flush()
        self.log('Saved file %s' % filename)