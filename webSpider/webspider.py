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
		req.encoding = 'GB2312'
		html = req.text
		div_bf = BeautifulSoup(html,'html.parser')
		div = div_bf.find_all('div',class_ = 'listmain')
		a_bf = BeautifulSoup(str(div[0]),'html.parser')
		a = a_bf.find_all('a')
		self.nums = len(a[12:])
		for each in a[12:]:
			self.names.append(each.string)
			self.urls.append(self.server + each.get('href'))
	def get_contents(self,target):
		req = requests.get(url=target)
		## change encoding mode..
		req.encoding = 'GB2312'
		html = req.text
		bf = BeautifulSoup(html,'html.parser')
		texts = bf.find_all('div',class_='showtxt')
		texts = texts[0].text.replace('\xa0'*8,'\n')
		return texts
	def writer(self,name,path,text):
		with open(path,'a',encoding='utf-8') as f:
			f.write(name + '\n')
			f.writelines(text)
			f.write('\n')
if __name__ == '__main__':
	server = 'http://www.biqukan.com/'
	url = 'http://www.biqukan.com/1_1094/'
	headers = {'user-agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.117 Safari/537.36'}
	
	text_downloader = downloader(server,url)
	text_downloader.get_download_url(headers)
	#print(f'content list {text_downloader.names}')
	print('Downloading......')
	for i in tqdm(range(text_downloader.nums),desc="Downloading noval..."):
		text_downloader.writer(text_downloader.names[i],'一念永恒.txt',text_downloader.get_contents(text_downloader.urls[i]))

	
	
	
	
	
