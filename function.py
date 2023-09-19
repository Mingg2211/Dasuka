import re
import string
from underthesea import word_tokenize
from rank_bm25 import BM25Okapi
from pyvi import ViTokenizer
STOPWORDS = ['hàng mới 100%']


def normalize_text(text):
    listpunctuation = string.punctuation.replace('%', '')
    for i in listpunctuation:
        text = text.replace(i, ' ')
    text = re.sub(' +', ' ', text)
    for stopword in STOPWORDS:
        if stopword in text :
            text = text.replace(stopword, '')
    text = word_tokenize(text, format='text')
    # text = text.replace(' ','_ ')
    return text.lower().strip()

def get8so(data_csv,query):
    filtered_df = data_csv[data_csv['hscode'].str.len() == 8]
    list_hs8so = list(filtered_df['mo_ta'])
    data = filtered_df.to_dict('records')
    nn = []
    for item in list_hs8so:
        item = item.lower()
        nn.append(ViTokenizer.tokenize(item))
    bm25 = BM25Okapi(nn)
    scores = bm25.get_scores(query)
    ranked_tariffs = list(enumerate(data))
    ranked_tariffs.sort(key=lambda x: scores[x[0]], reverse=True)
    return ranked_tariffs[:5]        

def create_text(row):
    return ' ||| '.join([str(row['Ma_HS']), str(row['Ten_SP']), str(row['Ngay_xac_nhan'])])
# print(normalize_text('.#&Hoá chất Axit nitric HNO3 68%, hàng mới 100%'))