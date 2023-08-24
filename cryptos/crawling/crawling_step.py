import os
import random
import requests
import pandas as pd
from datetime import datetime
import time
from queue import Queue
from threading import Thread

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from lxml.html import fromstring

from data_load.data_loading import LoadCrytpo


class Crawling(LoadCrytpo):
    
    def __init__(self, proxy=True, cores=1):

        LoadCrytpo.__init__(self)

        self.use_proxy = proxy
        
        self.current_proxy_index = 0
        self.saving_size = 50
        self.proxies = self.get_proxies()

        self.missed_urls = []

        if self.use_proxy:
            self.cores = cores 
        else:
            self.cores = cores

        print(f"NUMBER OF DRIVERS {self.cores}")
        
        os.environ["DIR_PATH"] = self.configs.load["resources"]["log_path"]
        self.queues = {"drivers": Queue(), "urls" :  Queue(), "results": Queue(), "base_path" : self.configs.load["resources"]["base_path"]}

    def get_proxies(self):

        url = 'https://sslproxies.org/'
        response = requests.get(url)
        parser = fromstring(response.text)
        proxies = []
        for i in parser.xpath('//tbody/tr')[:25]:
            if "minutes" in i.xpath('.//td[8]/text()')[0] or "seconds" in i.xpath('.//td[8]/text()')[0]:
                if i.xpath('.//td[7][contains(text(),"yes")]') and i.xpath('.//td[4]/text()')[0] in \
                    ["France", "Singapore", "Germany", "United States"]:

                    #Grabbing IP and corresponding PORT
                    proxy = ":".join([i.xpath('.//td[1]/text()')[0], i.xpath('.//td[2]/text()')[0]])
                    proxies.append({"ID" : proxy, 
                                    "TIME" : i.xpath('.//td[8]/text()')[0], 
                                    "COUNTRY": i.xpath('.//td[4]/text()')[0]})

        # random.shuffle(proxies)
        print(f"NUMBER OF PROXIES {len(proxies)}")

        return proxies
        
    def initialize_driver_firefox(self, proxy=True, prefs=False):
        """
        Initialize the web driver with Firefox driver as principal driver geckodriver
        parameters are here to not load images and keep the default css --> make page loading faster
        """

        if len(self.proxies)>0:
            PROXY =  self.proxies[self.current_proxy_index]["ID"]
        else:
            print("NO PROXY AVAILABLE!! ")
            self.use_proxy = False

        firefox_profile = webdriver.FirefoxProfile()

        if prefs:
            firefox_profile.set_preference('permissions.default.stylesheet', 2)
            firefox_profile.set_preference('permissions.default.image', 2)
            firefox_profile.set_preference('dom.ipc.plugins.enabled.libflashplayer.so', 'false')
            firefox_profile.set_preference('disk-cache-size', 8000)
            firefox_profile.set_preference("http.response.timeout", 300)
            firefox_profile.set_preference("dom.disable_open_during_load", True)

        if self.use_proxy:
            firefox_profile.set_preference("network.proxy.type", 1)
            firefox_profile.set_preference("network.proxy.http", PROXY.split(":")[0])
            firefox_profile.set_preference("network.proxy.http_port", PROXY.split(":")[1])
            self.current_proxy_index = (self.current_proxy_index +1)%len(self.proxies)

                        
            firefox_capabilities = webdriver.DesiredCapabilities.FIREFOX
            firefox_capabilities['marionette'] = True

            firefox_capabilities['proxy'] = {
                "proxyType": "MANUAL",
                "httpProxy": PROXY,
                "ftpProxy": PROXY,
                "sslProxy": PROXY
            }

        driver = webdriver.Firefox(capabilities=firefox_capabilities, log_path= os.environ["DIR_PATH"] + "/geckodriver.log") 
        driver.delete_all_cookies()
        driver.set_page_load_timeout(300) 

        if self.use_proxy:
            print(f"New driver = {self.proxies[self.current_proxy_index]['COUNTRY']}")
            self.current_proxy_index = (self.current_proxy_index +1)%len(self.proxies)

        return driver
    
    def initialize_driver_chrome(self, proxy=True, prefs=True):
        """
        Initialize the web driver with chrome driver as principal driver chromedriver.exe, headless means no open web page. But seems slower than firefox driver  
        parameters are here to not load images and keep the default css --> make page loading faster
        """

        if len(self.proxies)>0:
            PROXY =  self.proxies[self.current_proxy_index]["ID"]
        else:
            print("NO PROXY AVAILABLE!! ")
            self.use_proxy = False
        
        options = Options()
        if prefs:
            prefs = {
                    "profile.managed_default_content_settings.images":2,
                    'disk-cache-size': 8000,
                     "profile.default_content_setting_values.notifications":2,
                     "profile.managed_default_content_settings.stylesheets":2,
                     "profile.managed_default_content_settings.cookies":2,
                     "profile.managed_default_content_settings.javascript":2,
                     "profile.managed_default_content_settings.plugins":2,
                     "profile.managed_default_content_settings.popups":2,
                     "profile.managed_default_content_settings.geolocation":2,
                     "profile.managed_default_content_settings.media_stream":2,
                    }
            
            options.add_experimental_option("prefs", prefs)
            # options.add_argument("--headless") # Runs Chrome in headless mode.
            options.add_argument("--incognito")
            options.add_argument('--no-sandbox') # Bypass OS security model
            options.add_argument('--disable-gpu')  # applicable to windows os only
            # options.add_argument('start-maximized') 

        options.add_argument('disable-infobars')
        options.add_argument("--disable-extensions")
        options.add_argument("--enable-javascript")

        if self.use_proxy:
            options.add_argument('--proxy-server=%s' % PROXY)
            self.current_proxy_index = (self.current_proxy_index +1)%len(self.proxies)

        # service_args =["--verbose", "--log-path={0}".format(self.configs.load["resources"]["log_path"])]

        driver = webdriver.Chrome(executable_path=self.configs.load["resources"]["driver_path"], 
                                  chrome_options=options)
        driver.delete_all_cookies()
        driver.set_page_load_timeout(300) 

        if self.use_proxy:
            print(f"New driver = {self.proxies[self.current_proxy_index]['COUNTRY']}")
            self.current_proxy_index = (self.current_proxy_index +1)%len(self.proxies)

        return driver

    def delete_driver(self, driver):
        driver.close()
    
    def restart_driver(self, driver):

        try:
            self.delete_driver(driver)
        except Exception:
            print("ALREADY DELETED")
            pass

        if self.current_proxy_index == len(self.proxies):
            self.proxies = self.get_proxies()
            self.current_proxy_index = 0

        driver = self.initialize_driver_chrome()
        print(f"DRIVER {self.current_proxy_index}")

        return driver

    def initialize_queue_drivers(self):
        print(self.cores)
        for i in range(self.cores):
             self.queues["drivers"].put(self.initialize_driver_chrome())
        print(f"DRIVER QUEUE INITIALIZED WITH {self.cores} drivers {self.current_proxy_index}")
    
    def initialize_queue_urls(self, urls=[]):
        for url in urls:
             self.queues["urls"].put(url)

    def close_queue_drivers(self):
        for i in range(self.queues["drivers"].qsize()):
            driver = self.queues["drivers"].get()
            driver.close()
    
    def start_threads_and_queues(self, function, sub_loc=""):
        for _ in range(self.cores):
            t = Thread(target= self.queue_calls, args=(function, self.queues, sub_loc, ))
            t.daemon = True
            t.start()

    def get_url(self, driver, url):

        try:
            time.sleep(random.uniform(0.5,1.5))
            driver.get(url)

        except Exception:
            pass
        
        return driver
    
    def saving_queue_result(self, queues, sub_loc):

        results = []
        while queues["results"].qsize() > 0:
            results.append(queues["results"].get())

        # saving 
        hour = datetime.today().strftime("%Y-%m-%d %H-%M")
        if not os.path.isdir(sub_loc):
            os.mkdir(sub_loc)
        pd.DataFrame(results).to_csv(sub_loc + f"/{hour}.csv", index=False, sep=",")

    def queue_calls(self, function, queues, sub_loc):
        
        queue_url = queues["urls"]
        missed_urls = []
        
        #### extract all articles
        while True:
            driver = queues["drivers"].get()
            item = queue_url.get()

            try:
                driver = self.get_url(driver, item["url"])
                time.sleep(1)    
                
                try:
                    item["result"] = function(driver)
                    queues["results"].put(item)
                except Exception as e:
                    missed_urls.append(item["url"])
                    pass

                queues["drivers"].put(driver)
                queue_url.task_done()
                print(f"CRAWLED URL {item['url']}")

                if queues["results"].qsize() >= self.saving_size:
                    self.saving_queue_result(queues, sub_loc)
            
            except Exception as e:
                print(item['url'], e)
                driver = self.restart_driver(driver)
                
                missed_urls.append(item['url'])
                queue_url.task_done()
                queues["drivers"].put(driver)

            self.missed_urls = missed_urls