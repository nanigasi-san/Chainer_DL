#テキストデータのダウンロードをするスクリプト
import requests

def download_data(filename):
    res = requests.get("https://raw.githubusercontent.com/tomsercu/lstm/master/data/"+filename)
    with open("word2vec/data/"+filename,"w") as f:
        f.write(res.text)

download_data("ptb.train.txt")
download_data("ptb.valid.txt")
download_data("ptb.test.txt")
