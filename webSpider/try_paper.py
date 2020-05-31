import requests

if __name__ == '__main__':
    url = 'https://www.ixueshu.com/document/18debfaea2340c5d68d71b7aa22c4d32318947a18e7f9386.html'
    headers = {'user-agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.117 Safari/537.36'}
	
    req = requests.get(url,headers=headers)
    print(req.apparent_encoding)
    print(req.text)