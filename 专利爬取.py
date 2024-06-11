# -*- codeing = utf-8 -*-
# @Sofaware : PyCharm

import requests
from lxml import etree
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
#import xlwt
import csv
import os
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium import webdriver

#  123.0.6312.86
#  IPC主分类、发明名称、摘要、类名

if __name__ == '__main__':
    comman_names = ['G03F','G03G','G03H','G04G','G04R','G05F','G06N','G06Q','G06T','G06V','G07B','G07C','G07F','G07G','G08C','G08G'
,'G09B','G09F','G09G','G10L','G11B','G11C','G16B','G16C','G16H','G16Z','H01C','H01G','H01J','H01P','H03B','H03C','H03D','H03F','H03G','H03H','H03J','H03K','H03L','H03M','H04H',
'H04J','H04K','H04L','H04M','H04N','H04Q','H05H','A01F','B01J','B02C','B08B','B60L','B60W','C01B','C01G','C02F','C03B','C03C','C04B','C08K','C08L','C09D','C10B','C10G','C10J',
'C10L','C23C','C25D','C30B','D01F','E02B','E02D','E04B','E04G','E04H','F01D','F02B','F02C','F02F','F02G','F02M','F02N','F02P','F03B','F03D','F03G','F17C','F21L','F21V','F22B',
'F22D','F23B','F23C','F23D','F23G','F23K','F23L','F24F','F24H','F24S','F25B','F27B','F27D','F28B','F28D','F28F','G01N','G01T','G21B','G21C','G21D','G21F','G21G','H01F','H01H',
'H02B','H02G','H02H','H02K','H02N','H02P','H02S','H05B'
]
    print(len(comman_names))
    for comman_name in comman_names:
        #csv_file_path = f'{comman_name}.csv'  # 定义你想要保存数据的 CSV 文件名和路径
        #comman_name = "your_common_name"  # 假设这是你的文件名变量
        csv_file_path = r'E:\third work\数据集\{}.csv'.format(comman_name)  # 使用 format 方法进行字符串格式化
        #csv_file_path = r'E:\third work\数据集\all-zhuanli.csv'.format(comman_name)
        # 检查文件是否存在，如果不存在则添加表头
        if not os.path.exists(csv_file_path):
            with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['IPC主分类', '发明名称', '摘要'])  # 根据你的要求写入表头

        technologies = {
            "G10L": "语音分析或合成；语音识别；语音或声音处理；语音或音频编码或解码",
            "G11B": "基于记录载体和换能器之间的相对运动而实现的信息存储",
            "G11C": "静态存储器",
            "G16B": "生物信息学，例如特别适用于计算分子生物学中的遗传或蛋白质相关数据处理的信息与通信技术",
            "G16C": "计算化学;化学信息学;计算材料科学",
            "G16H": "医疗保健信息学，即专门用于处置或处理医疗或健康数据的信息和通信技术",
            "G16Z": "未列入其他类目的特别适用于特定应用领域的信息和通信技术"
        }

        opt = Options()
        # opt.add_argument("--headless")
        opt.add_argument("--disable-gpu")
        opt.add_experimental_option('excludeSwitches', ['enable - logging'])
        #chrome_driver_path = r'D:\chrome\chromedriver-win64\chromedriver-win64.exe'
        #executable_path = r'D:\桌面\configs\chromedriver123.exe'
        # 实例化 Chrome WebDriver，并将 ChromeOptions 对象传递给构造函数
        #web = webdriver.Chrome(options=opt, executable_path=chrome_driver_path)
        web = webdriver.Chrome(options=opt,
                     executable_path=r'D:\chrome\chromedriver-win64\chromedriver-win64.exe')  # 指定路径下的chrmedriver文件
        web.get("https://www.patentstar.com.cn/Search/TableSearch")

        web.find_element(By.ID, "Txt6").send_keys(comman_name)
        web.find_element(By.ID, "searchbtn").click()

        #web.find_element(By.ID, "loginname").send_keys("mtl13072961011")
        #web.find_element(By.ID, "password").send_keys("Iammtl20021011")
        web.find_element(By.ID, "loginname").send_keys("13571023373")
        web.find_element(By.ID, "password").send_keys("qll011227")
        web.find_element(By.ID, "login").click()
        time.sleep(5)
        with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for page in range(1,121):       # 1,201
                print(f'{comman_name}' + '正在爬第 ' + str(page))
                # 定位到页码输入框
                page_num_input = web.find_element(By.CLASS_NAME, "page_num")

                # 清除输入框中的当前内容
                page_num_input.clear()
                # 输入新的页码
                page_num_input.send_keys(page)
                web.find_element(By.CLASS_NAME, "page_btn").click()
                time.sleep(5)
                data_list = []
                for i in range(1,6):
                    xpath = f'//*[@id="listcontainer"]/div[{i}]/div/div[1]/label'
                    name = web.find_element(By.XPATH, xpath).text
                    try:
                        xpath = f'//*[@id="listcontainer"]/div[{i}]/div/div[2]/div[2]/div[4]/p/span'
                        content = web.find_element(By.XPATH, xpath).text
                    except NoSuchElementException:
                        try:
                            xpath =  f'//*[@id="listcontainer"]/div[{i}]/div/div[2]/div[2]/div[3]/p/span'
                            content = web.find_element(By.XPATH, xpath).text
                        except NoSuchElementException:
                            content = None  # 如果两种 XPath 都找不到元素，为content赋值为None

                    try:
                        xpath =f'//*[@id="listcontainer"]/div[{i}]/div/div[2]/div[2]/div[2]/p[3]/span/a/span'
                        ipc_dec = web.find_element(By.XPATH, xpath).text
                    except NoSuchElementException:
                        try:
                            xpath = f'//*[@id="listcontainer"]/div[{i}]/div/div[2]/div[2]/div[2]/p[1]/span/a/span'
                            ipc_dec = web.find_element(By.XPATH, xpath).text
                        except NoSuchElementException:
                            ipc_dec = None


                    data_list.append([ipc_dec, name, content])  # 按照你指定的顺序添加数据
                for data in data_list:
                    writer.writerow(data)



