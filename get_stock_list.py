from bs4 import BeautifulSoup
import requests
import re
from random import sample

"""
these functions retrieve the names of stocks listed on the FTSE 100 index from the yahoo finance website
"""

##################################
## list of stocks


w = requests.get('https://uk.finance.yahoo.com/q/cp?s=%5EFTSE')
w1 = requests.get('https://uk.finance.yahoo.com/q/cp?s=%5EFTSE&c=1')
w2 = requests.get('https://uk.finance.yahoo.com/q/cp?s=%5EFTSE&c=2')
wdj = requests.get('https://finance.yahoo.com/quote/%5EDJI/components?p=%5EDJI')
soup = BeautifulSoup(w.text, 'lxml')
soup1 = BeautifulSoup(w1.text, 'lxml')
soup2 = BeautifulSoup(w2.text, 'lxml')
soupdj = BeautifulSoup(wdj.text, 'html5lib')
p = re.compile('/q\?s=(\w+.\w*)')
pdj = re.compile('\?p=(\w+.\w*)')
table = soup.find("table", { "class" : "yfnc_tableout1" })
table1 = soup1.find("table", { "class" : "yfnc_tableout1" })
table2 = soup2.find("table", { "class" : "yfnc_tableout1" })
tabledj = soupdj.find("table", {"class" :"W(100%).M(0).BdB.Bdc($lightGray)"})

def get_stock_names(table):
    stocks = []
    for row in table.find_all("tr"):
        try:
            stock = re.search(p, str(row)).group(1)
            stocks.append(stock)
        except Exception: pass
    return stocks

def get_stocks():
    try:
        with open('stocks.txt', 'r') as f:
            tmp = f.read()
            assert len(tmp) > 10
            stocks = tmp.split()
            print("retrieving local stocks list")
        f.close()

    except Exception:
        print("retrieving stocks list from yahoo finance")
        tables = [table, table1, table2]
        stocks = [get_stock_names(t) for t in tables]
        stocks = [stock for tranche in stocks for stock in tranche]

        with open('stocks.txt', 'w') as f:
            for s in stocks:
                f.write(str(s))
                f.write(' ')
        f.close()
    return stocks

print(sample(get_stocks(), 10))

