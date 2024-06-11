# -*- coding=utf-8 -*-
# @Software: PyCharm

import os
import csv
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException

#
class PatentStarScraper:
    def __init__(self, common_name):
        self.common_name = common_name
        self.csv_file_path = f'{common_name}.csv'
        self.technologies = {
            "G10L": "语音分析或合成；语音识别；语音或声音处理；语音或音频编码或解码",
            "G11B": "基于记录载体和换能器之间的相对运动而实现的信息存储",
            "G11C": "静态存储器",
            "G16B": "生物信息学，例如特别适用于计算分子生物学中的遗传或蛋白质相关数据处理的信息与通信技术",
            "G16C": "计算化学;化学信息学;计算材料科学",
            "G16H": "医疗保健信息学，即专门用于处置或处理医疗或健康数据的信息和通信技术",
            "G16Z": "未列入其他类目的特别适用于特定应用领域的信息和通信技术"
        }
        self.setup_driver()
        self.setup_csv_file()

    def setup_driver(self):
        opt = Options()
        opt.add_argument("--disable-gpu")
        opt.add_experimental_option('excludeSwitches', ['enable-logging'])
        self.driver = webdriver.Chrome(options=opt, executable_path=r'D:\桌面\configs\chromedriver123.exe')
        self.driver.get("https://www.patentstar.com.cn/Search/TableSearch")

    def setup_csv_file(self):
        if not os.path.exists(self.csv_file_path):
            with open(self.csv_file_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['IPC主分类', '发明名称', '摘要'])

    def login(self):
        self.driver.find_element_by_xpath(f'// *[ @ id = "Txt6"]').send_keys(f"{self.common_name}")
        self.driver.find_element_by_xpath(f'// *[ @ id = "searchbtn"]').click()
        self.driver.find_element_by_xpath(f'// *[ @ id = "loginname"]').send_keys("mtl13072961011") #
        self.driver.find_element_by_xpath(f'// *[ @ id = "password"]').send_keys("Iammtl20021011") #
        self.driver.find_element_by_xpath(f'// *[ @ id = "login"]').click()
        time.sleep(5) # 风控 封的IP 显性等待

    def scrape_data(self):
        with open(self.csv_file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for page in range(1, 201):
                print(f'{self.common_name} 正在爬第 {page} 页')
                page_num_input = self.driver.find_element_by_class_name("page_num")
                page_num_input.clear()
                page_num_input.send_keys(page)
                self.driver.find_element_by_class_name("page_btn").click()
                time.sleep(5)
                data_list = self.extract_data_from_page()
                for data in data_list:
                    writer.writerow(data)

    def extract_data_from_page(self):
        data_list = []
        for i in range(1, 6):
            name = self.driver.find_element_by_xpath(f'//*[@id="listcontainer"]/div[{i}]/div/div[1]/label').text
            content = self.get_content(i)
            ipc_dec = self.get_ipc_dec(i)
            data_list.append([ipc_dec, name, content])
        return data_list

    def get_content(self, i):
        try:
            content = self.driver.find_element_by_xpath(
                f'//*[@id="listcontainer"]/div[{i}]/div/div[2]/div[2]/div[4]/p/span').text
        except NoSuchElementException:
            try:
                content = self.driver.find_element_by_xpath(
                    f'//*[@id="listcontainer"]/div[{i}]/div/div[2]/div[2]/div[3]/p/span').text
            except NoSuchElementException:
                content = None
        return content

    def get_ipc_dec(self, i):
        try:
            ipc_dec = self.driver.find_element_by_xpath(
                f'//*[@id="listcontainer"]/div[{i}]/div/div[2]/div[2]/div[2]/p[3]/span/a/span').text
        except NoSuchElementException:
            try:
                ipc_dec = self.driver.find_element_by_xpath(
                    f'//*[@id="listcontainer"]/div[{i}]/div/div[2]/div[2]/div[2]/p[1]/span/a/span').text
            except NoSuchElementException:
                ipc_dec = None
        return ipc_dec

    def run(self):
        self.login()
        self.scrape_data()
        self.driver.quit()

if __name__ == '__main__':
    common_name = 'G11B'
    scraper = PatentStarScraper(common_name)
    scraper.run()
