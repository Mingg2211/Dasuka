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
import nltk
from nltk.translate.bleu_score import sentence_bleu

class My_Search():
    tree_path = 'tree/tree.json'
    bm25_4so_path = 'bm25/4so/bm25_4so_model.pkl'
    bm25_quakhu_path = 'bm25/quakhu/bm25_quakhu_model.pkl'
    documents_quakhu_path = 'bm25/quakhu/documents_quakhu.pkl'
    bm25_caselaw_path = 'bm25/caselaw/bm25_caselaw_model.pkl'
    documents_caselaw_path = 'bm25/caselaw/documents_caselaw.pkl'
    def __init__(self, csv_bieuthue, csv_hs_quakhu, csv_caselaw):
        self.csv_bieuthue = csv_bieuthue
        self.csv_hs_quakhu = csv_hs_quakhu
        self.csv_caselaw = csv_caselaw
        self.pandas_quaku = pd.read_csv(self.csv_hs_quakhu, dtype={'Mã HS':str, 'Tên hàng':str, 'Ngày ĐK':str, 'Mã doanh nghiệp':str, 'Tên doanh nghiệp':str})
        self.pandas_caselaw = pd.read_csv(self.csv_caselaw, dtype={'Mã HS':str, 'Tên hàng':str, 'Ngày ĐK':str, 'Mã doanh nghiệp':str, 'Tên doanh nghiệp':str})
        self.root = None
        self.documents_qk = None
        self.documents_caselaw = None
        self.bm25_quakhu = None
        self.bm25_caselaw = None
            
    
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
            dashes = item['mo_ta'].count('- ')
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
        # ancestors.reverse()
        return [anc.name for anc in ancestors]
    def get_descendants_by_hscode(self,hscode):
        target_node = next((node for node in self.root.descendants if node.name['hscode'] == hscode), None)
        descendants = list(target_node.descendants)
        # descendants.reverse()
        return [des.name for des in descendants]
    def get_all_relative_by_hscode(self, hscode):
        ancestors = self.get_ancestors_by_hscode(hscode)
        descendants = self.get_descendants_by_hscode(hscode)
        ancestors.append(self.search_Tree(hscode))
        relative = ancestors + descendants
        return relative
    def search_Tree(self, hscode):
        target_node = next((node for node in self.root.descendants if node.name['hscode'] == hscode), None)
        return target_node.name
    def build_bm25_quakhu(self):
        df = self.pandas_quaku
        df['Tên hàng'] = df['Tên hàng'].apply(normalize_text)
        df['text'] = df.apply(create_text, axis=1)
        documents_qk = df['text'].tolist()
        tokenized_documents = [document.split() for document in documents_qk]
        bm25_qk = BM25Okapi(tokenized_documents)
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
        print('your query : ', query)
        tokenized_query = query.split()
        scores = self.bm25_quakhu.get_scores(tokenized_query)
        top_candidates = sorted(range(len(scores)), key=lambda i: -scores[i])[:5]
        result_qk = []
        for i in top_candidates:
            print(f"Document {i + 1}: {self.documents_qk[i]}, Score: {scores[i]}")
            if scores[i] > 10:
                result_qk.append(str(self.documents_qk[i]).split(' ||| ')[0])
        return result_qk
    def build_bm25_caselaw(self):
        df = self.pandas_caselaw
        # print(df.info())
        # print(df.head(5))
        df['mo_ta'] = df['mo_ta'].apply(normalize_text)
        df['text'] = df.apply(lambda row: ' ||| '.join(map(str, row)), axis=1)
        documents_caselaw = df['text'].tolist()
        # print(documents_caselaw)
        tokenized_documents = [document.split() for document in documents_caselaw]
        bm25_caselaw = BM25Okapi(tokenized_documents)
        return bm25_caselaw, documents_caselaw
    def load_bm25_caselaw(self):
        if os.path.exists(self.bm25_caselaw_path) and os.path.exists(self.documents_caselaw_path):
            print("Loading BM25_Caselaw ...")
            self.bm25_caselaw = joblib.load(self.bm25_caselaw_path)
            self.documents_caselaw = joblib.load(self.documents_caselaw_path)
            print("Done loading BM5_Caselaw !!!")
        else : 
            # save bm25_full to pkl 
            print("Building BM25_Caselaw ...")
            self.bm25_caselaw, self.documents_caselaw = self.build_bm25_caselaw()
            joblib.dump(self.bm25_caselaw, self.bm25_caselaw_path)
            joblib.dump(self.documents_caselaw, self.documents_caselaw_path)
            print("Done building BM25_Caselaw !!!")
        return self.bm25_caselaw, self.documents_caselaw
    def search_Caselaw(self, mo_ta:str):
        query = normalize_text(mo_ta)
        print('your query : ', query)
        tokenized_query = query.split()
        scores = self.bm25_caselaw.get_scores(tokenized_query)
        top_candidates = sorted(range(len(scores)), key=lambda i: -scores[i])[:5]
        result_caselaw = []
        for i in top_candidates:
            print(f"Document {i + 1}: {self.documents_caselaw[i]}, Score: {scores[i]}")
            result_caselaw.append(str(self.documents_caselaw[i]).split(' ||| ')[0])
        return result_caselaw
    def compute_bleu(self,mo_ta:str, list_tenhang:list):
        mat_hang_with_distance = []
        mo_ta = mo_ta.split()
        for tenhang in list_tenhang:
            tenhang = str(tenhang).split()
            distance = sentence_bleu(references=[mo_ta], hypothesis=tenhang, weights=(1, 0, 0, 0))
            mat_hang_with_distance.append( distance)
        return mat_hang_with_distance
    def search_pandas(self,hscode,mo_ta:str):
        #--------------------- Qúa khứ của company ---------------------#
        row_company = self.pandas_quaku[self.pandas_quaku['Mã HS'] == hscode][['Ngày ĐK', 'Mã HS', 'Tên hàng', 'Mã doanh nghiệp', 'Tên doanh nghiệp']]
        list_tenhang = row_company['Tên hàng'].to_list()
        list_distance = self.compute_bleu(mo_ta, list_tenhang)
        row_company = row_company.assign(distance=list_distance,)
        row_company = row_company.sort_values(by=['distance'], ascending=False)[:3]
        row_company = row_company.drop(columns=['distance'])
        list_qk_company = row_company.to_dict('records')
        #--------------------- Qúa khứ của caselaw ---------------------#
        row_caselaw = self.pandas_caselaw[self.pandas_caselaw['Mã HS'] == hscode]
        list_tenhang = row_caselaw['Tên hàng'].to_list()
        list_distance = self.compute_bleu(mo_ta, list_tenhang)
        row_caselaw = row_caselaw.assign(distance=list_distance,)
        row_caselaw = row_caselaw.sort_values(by=['distance'], ascending=False)[:2]
        row_caselaw = row_caselaw.drop(columns=['distance'])
        list_qk_caselaw = row_caselaw.to_dict('records')
        
        result = list_qk_company + list_qk_caselaw
        return result
    
    def search_main(self, mo_ta:str):
        result_qk = self.search_QK(mo_ta) 
        result_caselaw = self.search_Caselaw(mo_ta)
        hs_candidates = []
        tmp_list = result_qk + result_caselaw
        for item in tmp_list:
            if item not in hs_candidates:
                hs_candidates.append(item)
        print(hs_candidates)
        # check bm25 . ti tat di
        # return hs_candidates
        # 
        result = []
        for item in hs_candidates:
            try:
                tree = self.search_Tree(item)
            except Exception as e:
                continue
            result.append(tree)
        # message = '\n'.join(result)
        # print('Tìm thấy {len_result} mã hscode phù hợp với mô tả "{mo_ta}" : \n'.format(len_result=len(result), mo_ta=mo_ta))
        # print(result)
        return result
    def search_documents_qk(self, hscode:str):
        hs_code = int(hscode)
        df = pd.read_csv(self.csv_hs_quakhu)
        df['Ma_HS'] = df['Ma_HS'].astype(int)
        rows = df[df['Ma_HS'] == hs_code]
        return rows.head(5)
        # return rows.to_dict('records')[:10]
    def get_all_bieu_thue(self):
        df = pd.read_csv(self.csv_bieuthue)
        result = df.to_dict('records')
        return result
        
# if __name__ == '__main__':
#     my_search = My_Search('data/bieu_thue.csv', 'company_data/Diep_Dasuka.csv', 'company_data/clone_caselaw.csv')
#     my_search.load_tree()
    # my_search.display_tree()
    # print(my_search.get_ancestors_by_hscode('39231090'))
    # print(my_search.get_descendants_by_hscode('7326'))
    # my_search.load_bm25_quakhu()
    # my_search.build_bm25_caselaw()
    # my_search.load_bm25_caselaw()
    # print(my_search.search_documents_qk('96190099'))
    # print(my_search.search_QK('máy cạo râu'))
    # print(my_search.search_Caselaw('thịt lợn đông lạnh'))
    # print(my_search.search_Tree('#46'))
    # while True:
    #     mo_ta = str(input("Nhập vào mô tả : "))
    #     if mo_ta != 'q':
    #         try:
    #             print(my_search.search_main(mo_ta))
    #         except Exception as e:
    #             print(e)
    #             continue
    #     else:
    #         break
    # df = pd.read_csv('company_data/Diep_Dasuka.csv')
    # mota_list = df['Tên hàng'].tolist()
    # for i in range(len(mota_list)):
        
            
    #     # print(mota_list[i])
    #     # print(my_search.search_main(mota_list[i]))
    #     ma_hs = str(df.iloc[i]['Mã HS'])
    #     # print(ma_hs)
    #     # for hscode_searched in my_search.search_main(mota_list[i]):
    #         # print(type(hscode_searched))
    #     if ma_hs in my_search.search_QK(mota_list[i]):
    #         mess = str(mota_list[i]) + ',' + str(my_search.search_QK(mota_list[i])) +',' +str(ma_hs) + ',true'
    #         print('true')
    #         with open('hs_diep_checker.txt','a', encoding='utf-8') as f:
    #             f.write(mess)
    #             f.write('\n')
    #     else:
    #         print('false')
    #         with open('hs_diep_checker.txt','a', encoding='utf-8') as f:
    #             mess = str(mota_list[i]) + ',' + str(my_search.search_QK(mota_list[i])) +',' +str(ma_hs) + ',false'
    #             f.write(mess)
    #             f.write('\n')
    # print(my_search.search_main('thịt lợn đông lạnh'))
    # print(my_search.search_pandas('02032900','thịt lợn đông lạnh'))
    # print(my_search.search_Tree('01022911'))