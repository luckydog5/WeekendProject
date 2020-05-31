import requests
from bs4 import BeautifulSoup


if __name__ == '__main__':
    url = 'https://www.cd50185f796b676b.xyz/shipin/play-96990.html'
    #'https://www.mmxzxl1.com/common/wm/2020_05/26/wm_CfLhggBH_wm.mp4'
    #'https://www.mmxzxl1.com/common/wm/2020_05/26/wm_CfLhggBH_wm.mp4'
    req = requests.get(url)
    req.encoding = req.apparent_encoding
    html = req.text 
    div_bf = BeautifulSoup(html,'html.parser')
    div = div_bf.find_all('div',class_='hy-layout active clearfix')
    print(div)
    #print(html)