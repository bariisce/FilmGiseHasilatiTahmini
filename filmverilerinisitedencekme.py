# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 01:00:30 2023

@author: Baris
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd

# EdgeDriver'ı başlatma
driver = webdriver.Edge()

# Web sayfasını açma
url = "https://www.the-numbers.com/market/2020/top-grossing-movies"
driver.get(url)

try:
    # page_filling_chart id'li div'i bulma
    div_element = driver.find_element(By.ID, "page_filling_chart")
    
    # table elementini bulma
    table_element = div_element.find_element(By.TAG_NAME, "table")
    
    # tbody elementini bulma
    tbody_element = table_element.find_element(By.TAG_NAME, "tbody")
    
    # tr elementlerini bulma
    tr_elements = tbody_element.find_elements(By.TAG_NAME, "tr")[1:]  # 1. tr hariç
    
    # Verileri pandas DataFrame'e ekleme
    data = []
    for tr_element in tr_elements:
        # td elementlerini bulma
        td_elements = tr_element.find_elements(By.TAG_NAME, "td")
        row_data = [td.text for td in td_elements]
        data.append(row_data)

    columns = ["Rank", "Movie", "Release Date", "Distributor", "Genre", "2023 Gross", "Tickets Sold"]
    df = pd.DataFrame(data, columns=columns)

    # DataFrame'i Excel dosyasına yazma
    df.to_excel("movie_data2020.xlsx", index=False)

finally:
    # WebDriver'ı kapatma
    driver.quit()