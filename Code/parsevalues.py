#import packages
import requests
import urllib.request
import pandas as pd 
from bs4 import BeautifulSoup
import time
from selenium import webdriver
import selenium.webdriver.chrome.service as servicea
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.by import By
from tqdm import tqdm

def webtotext(url, domain):
    try:
        res = requests.get(url)
    except ConnectionRefusedError:
    	print("connectionrefusederror")
    	return "400"
    if res.status_code == 404:
    	print("status code bad")
    	return "400"
    else:
        html_page = res.content
        soup = BeautifulSoup(html_page, 'html.parser')
        if domain == 'usatoday.com':
            text = soup.find_all("p" ,{"class":"gnt_ar_b_p"})
        elif domain == 'cnn.com':
            text = soup.find_all("p", {"data-component-name": "paragraph"})
            #title = soup.find("title")
        elif (domain == 'bbc.co.uk'):
            text = soup.find_all('div', {"class":["ssrcss-7uxr49-RichTextContainer e5tfeyi1"]})
        elif domain == 'nytimes.com':
            display = Display(visible=0, size=(800, 600))
            service = servicea.Service('/usr/local/bin/chromedriver')
            service.start()
            #capabilities = {'chrome.binary':'path/to/'}
            chrome_options = webdriver.ChromeOptions()
            chrome_options.add_argument("--disable-notifications")            
            driver = webdriver.Remote(service.service_url, desired_capabilities=DesiredCapabilities.CHROME, chrome_options=chrome_options)
            driver.get(url)
            elem = driver.find_elements(By.CLASS_NAME,"css-53u6y8")
            stringoftext = "".join([element.text for element in elem])
            time.sleep(2) # Let the user actually see something!
            driver.quit()
            display.stop()
            return stringoftext
        elif domain == 'breitbart.com':
            text = soup.find_all('div', {'class':'entry-content'})
        elif domain == 'huffingtonpost.com':
            text = soup.find_all('div', {'class':'primary-cli cli cli-text'})
        elif domain == 'wsj.com':
            text = soup.find_all('div', {'class':"wsj-snippet-body"})
        elif domain == 'washingtonpost.com':
            text = soup.find_all('div', {'class':'article-body'})
        elif domain == 'dailykos.com':
            #text = soup.find_all('div', {'class':'dkimg-c'})
            text = soup.find('div', {'class':'story-column'})
        elif domain == 'latimes':
            display = Display(visible=0, size=(800, 600))
            service = servicea.Service('/usr/local/bin/chromedriver')
            service.start()
            #capabilities = {'chrome.binary':'path/to/'}
            chrome_options = webdriver.ChromeOptions()
            chrome_options.add_argument("--disable-notifications")            
            driver = webdriver.Remote(service.service_url, desired_capabilities=DesiredCapabilities.CHROME, chrome_options=chrome_options)
            #driver = webdriver.Remote(service.service_url, desired_capabilities=DesiredCapabilities.CHROME)        
            driver.get(url)
            elem = driver.find_elements(By.XPATH,'@context":"http://schema.org","@type":"NewsArticle"')
            stringoftext = "".join([element.text for element in elem])
            time.sleep(2) # Let the user actually see something!
            driver.quit()
            display.stop()
            return stringoftext
        elif domain == 'foxnews.com':
            #display = Display(visible=0, size=(800, 600))
            service = servicea.Service('/usr/local/bin/chromedriver')
            service.start()
            #capabilities = {'chrome.binary':'path/to/'}
            chrome_options = webdriver.ChromeOptions()
            chrome_options.add_argument("--disable-notifications")            
            driver = webdriver.Remote(service.service_url, desired_capabilities=DesiredCapabilities.CHROME, chrome_options=chrome_options)
            #driver = webdriver.Remote(service.service_url, desired_capabilities=DesiredCapabilities.CHROME)        
            driver.get(url)
            elem = driver.find_elements(By.CLASS_NAME,"article-body")
            stringoftext = "".join([element.text for element in elem])
            time.sleep(2) # Let the user actually see something!
            driver.quit()
            #display.stop()
            return stringoftext
        else:
            text = "400"

        #print(type(text[0]), text[0])
        stringoftext = ''.join([tag.get_text() for tag in text])
        a = stringoftext.find("Daily Kos relies")
        if a != -1:
            stringoftext = stringoftext[:a]
        #title = title.get_text()
        #print("stringoftext \n", stringoftext, "\n type:", type(stringoftext))
    
    return stringoftext

dataset = pd.read_csv('/Users/sophiawalton/Documents/CS6501/newsArticlesWithLabels.tsv', sep = '\t')
print(dataset.head())
#article, title = webtotext("https://www.cnn.com/2013/06/28/politics/obama-contraceptives/index.html", "cnn.com")
#print(title,"\n" ,article)

#get the list of domains
urllist = list(dataset['url'])
print(urllist[0])
from urllib.parse import urlparse
domainlist = []

cnnn = 0
for row in urllist:
    t = urlparse(row).netloc
    #print(t)
    if t.find('.co.uk') != -1:
        domain = '.'.join(t.split('.')[-3:])
    else:
        domain = '.'.join(t.split('.')[-2:])
    if domain == 'cnn.com':
        cnnn+= 1
    domainlist.append(domain)
    #print(domain)
dataset['domain'] = domainlist
print(dataset.head())

dataset['domain'] = domainlist
possdomain = set(domainlist)
possdomain = list(possdomain)
possdomain = ['foxnews.com']


print(possdomain)

for domain in possdomain:
	thisdata  = dataset.groupby('domain').get_group(domain)
	print(thisdata)
	checklist = thisdata['url']
	textlist = []
	for index, row in tqdm(thisdata.iterrows(), desc=domain):
		try: 
			article = webtotext(row['url'], domain)
			textlist.append(article)
			#print(article)
		except KeyError:
			textlist.append("400")
			print(row[0], "keyerror")
		except Exception:
			textlist.append("400")
			print(row[0], "exception")
	thisdata['text'] = textlist
	domainname = domain.split('.')[0]
	thisdata.to_csv('/Users/sophiawalton/Documents/CS6501/newsarticleswtext'+domainname+'.tsv', sep='\t')
	print(domain, 'csv done')
