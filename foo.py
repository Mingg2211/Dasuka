import pandas as pd
import numpy as np
import json
from anytree import Node, RenderTree, PreOrderIter
from anytree.exporter import JsonExporter
from anytree.importer import JsonImporter
import os
from function import *
from rank_bm25 import BM25Okapi, BM25L, BM25Plus
import joblib

class My_Search():
    tree_path = 'tree/tree.json'
    bm25_4so_path = 'bm25/4so/bm25_4so_model.pkl'
    bm25_quakhu_path = 'bm25/quakhu/bm25_quakhu_model.pkl'
    documents_quakhu_path = 'bm25/quakhu/documents_quakhu.pkl'
    bm25_full_so_path = 'bm25/full_so/bm25_full_so_model.pkl'
    documents_full_so_path = 'bm25/full_so/documents_full_so.pkl'
    def __init__(self, csv_bieuthue, csv_hs_quakhu):
        self.csv_bieuthue = csv_bieuthue
        self.csv_hs_quakhu = csv_hs_quakhu
        self.root = None
        self.document_4_so = None
        self.documents_qk = None
        self.documents_full_so = None
        self.bm25_4so = None
        self.bm25_quakhu = None
        self.bm25_full_so = None
    def build_tree(self):
        # build tree and save
        df = pd.read_csv(self.csv_bieuthue)
        phan_chuong = df[df.hscode.isna() & ~df.mo_ta.str.contains('-')]
        nhom_hs = df.drop(phan_chuong.index)
        nhom_hs.hscode = nhom_hs.hscode.replace(np.nan, '')
        nhom_hs.hscode = nhom_hs.hscode.astype('str')
        nhom_hs.hscode = nhom_hs.hscode.apply(lambda x: x.replace('.',''))
        nhom_hs_dict = nhom_hs.to_dict('records')
        root = Node("Root")
        # Duyệt qua dữ liệu để xây dựng cây
        current_node = root
        for item in nhom_hs_dict:
            dashes = item['mo_ta'].count('-')
            name = item
            while len(current_node.ancestors) >= dashes + 1:
                current_node = current_node.parent
            new_node = Node(name, parent=current_node)
            current_node = new_node
        exporter = JsonExporter(indent=2, sort_keys=True, ensure_ascii=False)
        with open(self.tree_path, mode='w', encoding='utf-8') as f:
            exporter.write(root, f)
        return root
    def load_tree(self):
        if os.path.exists(self.tree_path):
            print("Loading Tree ...")
            importer = JsonImporter()
            with open(self.tree_path, 'r') as json_file:
                data = json_file.read()
            self.root = importer.import_(data)
            print("Done loading Tree !!!")
        else : 
            # save tree to json format
            print("Building Tree ...")
            self.root = self.build_tree()
            print("Done building Tree !!!")
        return self.root
    def display_tree(self):
        for pre, fill, node in RenderTree(self.root):
            print(f"{pre}{node.name}")     
    def get_ancestors_by_hscode(self,hscode):
        target_node = next((node for node in self.root.descendants if node.name['hscode'] == hscode), None)
        ancestors = list(target_node.ancestors)
        ancestors.reverse()
        return [anc.name for anc in ancestors]
    def get_descendants_by_hscode(self,hscode):
        target_node = next((node for node in self.root.descendants if node.name['hscode'] == hscode), None)
        descendants = list(target_node.descendants)
        # descendants.reverse()
        return [des.name for des in descendants] 
    def search_Tree(self, hscode):
        target_node = next((node for node in self.root.descendants if node.name['hscode'] == hscode), None)
        return target_node.name
    def build_bm25_quakhu(self):
        df = pd.read_csv(self.csv_hs_quakhu)
        df['Ten_SP'] = df['Ten_SP'].apply(normalize_text)
        df['text'] = df.apply(create_text, axis=1)
        documents_qk = df['text'].tolist()
        tokenized_documents = [document.split() for document in documents_qk]
        bm25_qk = BM25Plus(tokenized_documents)
        return bm25_qk, documents_qk
    def load_bm25_quakhu(self):
        if os.path.exists(self.bm25_quakhu_path) and os.path.exists(self.documents_quakhu_path):
            print("Loading BM25_QK ...")
            self.bm25_quakhu = joblib.load(self.bm25_quakhu_path)
            self.documents_qk = joblib.load(self.documents_quakhu_path)
            print("Done loading BM5_QK !!!")
        else : 
            # save bm25_qk to pkl 
            print("Building BM25_QK ...")
            self.bm25_quakhu, self.documents_qk = self.build_bm25_quakhu()
            joblib.dump(self.bm25_quakhu, self.bm25_quakhu_path)
            joblib.dump(self.documents_qk, self.documents_quakhu_path)
            print("Done building BM25_QK !!!")
        return self.bm25_quakhu, self.documents_qk    
    
    def search_QK(self, mo_ta:str):
        query = normalize_text(mo_ta)
        tokenized_query = query.split()
        scores = self.bm25_quakhu.get_scores(tokenized_query)
        top_candidates = sorted(range(len(scores)), key=lambda i: -scores[i])[:2]
        result_qk = []
        tham_khao_hs = []
        for i in top_candidates:
            # print(f"Document {i + 1}: {self.documents_qk[i]}, Score: {scores[i]}")
            tham_khao_hs.append(self.documents_qk[i])
            if scores[i] != 0:
                result_qk.append(str(self.documents_qk[i]).split(' ||| ')[0])
        return result_qk, tham_khao_hs
    def build_bm25_full_so(self):
        df = pd.read_csv(self.csv_bieuthue)
        df['mo_ta'] = df['mo_ta'].apply(normalize_text)
        df['text'] = df.apply(lambda row: ' ||| '.join(map(str, row)), axis=1)
        documents_full_so = df['text'].tolist()
        tokenized_documents = [document.split() for document in documents_full_so]
        bm25_full_so = BM25Plus(tokenized_documents)
        return bm25_full_so, documents_full_so
    def load_bm25_full_so(self):
        if os.path.exists(self.bm25_full_so_path) and os.path.exists(self.documents_full_so_path):
            print("Loading BM25_Full ...")
            self.bm25_full_so = joblib.load(self.bm25_full_so_path)
            self.documents_full_so = joblib.load(self.documents_full_so_path)
            print("Done loading BM5_Full !!!")
        else : 
            # save bm25_full to pkl 
            print("Building BM25_Full ...")
            self.bm25_full_so, self.documents_full_so = self.build_bm25_full_so()
            joblib.dump(self.bm25_full_so, self.bm25_full_so_path)
            joblib.dump(self.documents_full_so, self.documents_full_so_path)
            print("Done building BM25_Full !!!")
        return self.bm25_full_so, self.documents_full_so
    def search_Full(self, mo_ta:str):
        query = normalize_text(mo_ta)
        tokenized_query = query.split()
        scores = self.bm25_full_so.get_scores(tokenized_query)
        top_candidates = sorted(range(len(scores)), key=lambda i: -scores[i])[:3]
        result_full = []
        for i in top_candidates:
            # print(f"Document {i + 1}: {self.documents_full_so[i]}, Score: {scores[i]}")
            result_full.append(str(self.documents_full_so[i]).split(' ||| ')[0])
        return result_full
    def search_main(self, mo_ta:str):
        result_qk = self.search_QK(mo_ta) 
        result_full = self.search_Full(mo_ta)
        hs_candidates = []
        tmp_list = result_qk + result_full
        for item in tmp_list:
            if item not in hs_candidates:
                hs_candidates.append(item)
        print(hs_candidates)
        result = []
        for item in hs_candidates:
            result.append(str(self.search_Tree(item)))
        message = '\n'.join(result)
        print('Tìm thấy {len_result} mã hscode phù hợp với mô tả "{mo_ta}" : \n'.format(len_result=len(result), mo_ta=mo_ta) + message)
        return result
if __name__ == '__main__':
    my_search = My_Search('data/data_sample.csv', 'data/HSCode_sample.csv')
    my_search.load_tree()
    # my_search.display_tree()
    print(my_search.get_ancestors_by_hscode('39231090'))
    # print(my_search.get_descendants_by_hscode('7326'))
    # my_search.load_bm25_quakhu()
    # my_search.load_bm25_full_so()
    # print(my_search.search_QK('máy cạo râu'))
    # print(my_search.search_Full('thịt lợn đông lạnh'))
    # print(my_search.search_Tree('#46'))
    # while True:
    #     mo_ta = str(input("Nhập vào mô tả : "))
    #     if mo_ta != 'q':
    #         my_search.search_main(mo_ta)
    #     else:
    #         break
        