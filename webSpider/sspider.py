## project webspider
## downloading something interesting.
## amazing text with execiting contents.
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
class downloader(object):   
    def __init__(self,server,url):
        self.url = url 
        self.server = server 
        self.names = []
        self.urls = []
        self.nums = 0   
    def get_download_url(self,headers):
        req = requests.get(url = self.url,headers=headers)
        req.encoding = req.apparent_encoding
        html = req.text
        div_bf = BeautifulSoup(html,'html.parser')
        div = div_bf.find_all('div',class_ = 'text-list-html')
        a_bf = BeautifulSoup(str(div[0]),'html.parser')
        a = a_bf.find_all('a')
        self.nums = len(a[:-11])
        for each in a[:-11]:
            print(f"each href {each.get('href')}")
            print(f"each name {each.get('title')}")
            self.names.append(each.get('title'))
            self.urls.append(self.server + each.get('href'))
        print(self.names,self.urls)
    def get_contents(self,target):
        req = requests.get(url=target)
        ## change encoding mode..
        req.encoding = req.apparent_encoding
        html = req.text
        bf = BeautifulSoup(html,'html.parser')
        texts = bf.find_all('div',class_='content')
        #texts = texts[0].text.replace('\xa0'*8,'\n')
        texts = texts[0].text
        return texts
    def writer(self,name,path,text):
        with open(path,'a',encoding='utf-8') as f:
            f.write(name + '\n')
            f.write('\n')
            f.writelines(text)
            f.write('\n')
            f.write('\n')
if __name__ == '__main__':
    server = 'https://www.951cf.com/'
    #url = 'https://www.hhf627.com/xiaoshuo/list-%E5%AE%B6%E5%BA%AD%E4%B9%B1%E4%BC%A6-35.html'
    
    headers = {'user-agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.117 Safari/537.36'}

    #text_downloader = downloader(server,url)
    #text_downloader.get_download_url(headers)
    print('Downloading......')
    for n in range(1,2):
        url = 'https://www.951cf.com/xiaoshuo/list-%E5%AE%B6%E5%BA%AD%E4%B9%B1%E4%BC%A6-' + str(n) + '.html' 
        text_downloader = downloader(server,url)
        text_downloader.get_download_url(headers)
        path = 'page' + str(n) + '.txt'
        try:
            for i in tqdm(range(text_downloader.nums),desc="Downloading noval..."):

                text_downloader.writer(text_downloader.names[i],path,text_downloader.get_contents(text_downloader.urls[i]))

        except Exception as e:
            print(e)
            continue
        
	
	
	
