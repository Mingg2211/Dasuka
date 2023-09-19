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
    def __init__(self, csv_bieuthue, csv_hs_quakhu):
        self.csv_bieuthue = csv_bieuthue
        self.csv_hs_quakhu = csv_hs_quakhu
        self.tree_path = 'tree/tree.json'
        self.bm25_4so_path = 'bm25/4so/bm25_4so_model.pkl'
        self.bm25_quakhu_path = 'bm25/quakhu/bm25_quakhu_model.pkl'
        self.root = None
        self.document_4_so = self.build_tree_and_save()
        self.document_qk = self.build_tree_and_save()
        self.bm25_4so = None
        self.bm25_quakhu = None
    def build_tree_and_save(self):
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
        # build bm25 tren phan nhom
        phan_nhom_names = [node.name for node in list(root.children)]
        document_4_so = []
        for phan_nhom in phan_nhom_names:
            document_4_so.append(phan_nhom['hscode'] + ' *** '+ normalize_text(phan_nhom['mo_ta'])) 
        # phan_nhom = [item['mo_ta'] for item in phan_nhom_names]
        # self.mo_ta_4_so = [normalize_text(mo_ta) for mo_ta in phan_nhom]
        # tao bm25
        tokenized_documents = [document.split() for document in document_4_so]
        bm25 = BM25Plus(tokenized_documents)
        # save bm25
        joblib.dump(bm25, self.bm25_4so_path)
        # export tree to json format
        exporter = JsonExporter(indent=2, sort_keys=True, ensure_ascii=False)
        # tree_json = exporter.export(root)
        # print(tree_json)
        with open(self.tree_path, mode='w', encoding='utf-8') as f:
            exporter.write(root, f)
        quakhu_df = pd.read_csv(self.csv_hs_quakhu)
        quakhu_df['text'] = quakhu_df.apply(create_text, axis=1)
        document_qk = quakhu_df['text'].tolist()
        tokenized_documents_qk = [document.split() for document in document_qk]
        bm25_qk = BM25Plus(tokenized_documents_qk)
        joblib.dump(bm25_qk, self.bm25_quakhu_path)
        return document_4_so, document_qk
    def load_tree_and_bm25(self):
        if not os.path.exists(self.tree_path):
            print("---------- Cây đang trong qúa trình khởi tạo ----------")
            with open(self.tree_path, 'w') as fp:
                pass
            self.document_4_so = self.build_tree_and_save()
            print("---------- Khởi tạo cây thành công, lưu trữ tại {tree_path}".format(tree_path = self.tree_path))
        else :
            print("---------- Khởi tạo cây thành công ----------")
            importer = JsonImporter()
            with open(self.tree_path, 'r') as json_file:
                data = json_file.read()
            self.root = importer.import_(data)
        if not os.path.exists(self.bm25_4so_path):
            print("---------- Tiến hành khởi tạo BM25 Biểu thuế ----------")
        else:
            print("---------- Khởi tạo BM25 Biểu thuế thành công ----------")
            self.bm25_4so = joblib.load(self.bm25_4so_path)
        if not os.path.exists(self.bm25_quakhu_path):
            print("---------- Tiến hành khởi tạo BM25 Quá khứ ----------")
        else:
            print("---------- Khởi tạo BM25 Quá khứ thành công ----------")
            self.bm25_quakhu = joblib.load(self.bm25_quakhu_path)
            return self.root, self.bm25_4so, self.document_4_so, self.bm25_quakhu, self.document_qk
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
    def search_top_down(self, ten_sp):
        query = normalize_text(ten_sp)
        # print(query)
        tokenized_query = query.split()
        scores = self.bm25_4so.get_scores(tokenized_query)
        top_candidates = sorted(range(len(scores)), key=lambda i: -scores[i])[:5]
        for i in top_candidates:
            print(f"Document {i + 1}: {self.document_4_so[i]}, Score: {scores[i]}")
        return top_candidates
        # nen de mota 4 so dang hscode , mota de de truy xuat hscode

        hscode_candidates = []
        return hscode_candidates
    
    
    # TODO: code tim kiem top down (BFS + DFS), code tìm kiem bottom-up (bm25 8 so)
    def search_hs_quakhu(self, ten_sp):
        query = normalize_text(ten_sp)
        # print(query)
        tokenized_query = query.split()
        scores = self.bm25_quakhu.get_scores(tokenized_query)
        top_candidates = sorted(range(len(scores)), key=lambda i: -scores[i])[:3]
        for i in top_candidates:
            print(f"Document {i + 1}: {self.document_qk[i]}, Score: {scores[i]}")
        return top_candidates
    
    
if __name__ == '__main__':
    my_search = My_Search(csv_bieuthue='data/data_sample.csv', csv_hs_quakhu='data/HSCode_sample.csv')
    # print(my_search.build_tree_and_save())
    my_search.load_tree_and_bm25()
    to_tien = my_search.get_ancestors_by_hscode('01013010')
    print(to_tien)
    
    # while True:
    #     user_input = str(input('Nhập vaò mô tả san phẩm : '))
    #     if user_input != 'q':
    #         print(my_search.search_hs_quakhu(user_input))
    #     else:
    #         print("Thoát chương trình\n")
    #         break
    # my_search.display_tree()