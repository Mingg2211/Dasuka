from fastapi import FastAPI, UploadFile
import uvicorn
from typing import List, Optional, Union
from pydantic import BaseModel,Field
from datetime import datetime, timezone
from starlette.middleware.cors import CORSMiddleware
from typing import Optional
import uuid
from demo_dasuka import My_Search
import pandas as pd
import os
app = FastAPI()
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_methods=["*"],
                   allow_headers=["*"])

class QK(BaseModel):
    id:Optional[str]
    hscode: str
    description: str
    time: Optional[datetime]
    own:  str

@app.get("/")
async def hello():
    return {"Welcome to": " HScode Searcher"}

@app.post("/upload_csv")
async def upload_csv(file: UploadFile):
    file_path = os.path.join("company_data", file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    global my_search
    my_search = My_Search(csv_bieuthue='company_data/vippro.csv', csv_hs_quakhu=file_path, csv_caselaw='company_data/clone_caselaw.csv')
    my_search.load_tree()
    my_search.load_bm25_caselaw()
    my_search.load_bm25_quakhu()
    return {"response": "Done Upload {}".format(file.filename)}
    
@app.get("/search_main")
async def search_main(mo_ta:str):
    result = my_search.search_main(mo_ta)
    # message = '\n'.join(result)
    # return {"response": 'Tìm thấy {len_result} mã hscode phù hợp với mô tả "{mo_ta}" : \n'.format(len_result=len(result), mo_ta=mo_ta) + message}
    return {"response": result}
@app.get("/search_tree")
async def search_tree(hscode:str):
    result = my_search.search_Tree(hscode)
    return {"response": result}
@app.get("/get_ancestors_by_hscode")
async def get_ancestors_by_hscode(hscode:str):
    result = my_search.get_ancestors_by_hscode(hscode)
    return {"response": result}
@app.get("/get_descendants_by_hscode")
async def get_descendants_by_hscode(hscode:str):
    result = my_search.get_descendants_by_hscode(hscode)
    return {"response": result}
@app.get("/get_all_relative_by_hscode")
async def get_all_relative_by_hscode(hscode:str):
    result = my_search.get_all_relative_by_hscode(hscode)
    return {"response": result}
@app.get("/search_documents_qk")
async def search_documents_qk(hscode:str):
    result = my_search.search_documents_qk(hscode)
    return {"response": result}
@app.get("/search_pandas")
async def search_pandas(hscode:str,mo_ta:str):
    result = my_search.search_pandas(hscode,mo_ta)
    return result
@app.get("/get_all_bieu_thue")
async def get_all_bieu_thue():
    result = my_search.get_all_bieu_thue()
    return {"response": result}

@app.get("/getqkall", response_model=List[QK])
async def getallqk():
    df = pd.read_csv("data/qk.csv")

    # Convert DataFrame to a list of dictionaries
    qk_records = df.to_dict(orient='records')

    # Convert the records to a list of QK objects
    qk_list = [QK(**record) for record in qk_records]

    return qk_list
        
@app.post("/addqk")
async def add_qk(item: QK):
    # Load the existing data from the CSV file, if any
    df = pd.read_csv("data/qk.csv")
    print(str(item.hscode) in df['hscode'].values.astype(str))
    print(item.description in df['description'].values)
    # Check if the given hscode and description combination already exists
    if (str(item.hscode) in df['hscode'].values.astype(str)) and (item.description in df['description'].values):
        # If a matching record already exists, return an error message or handle it as needed.
        return {"message": "QK record with the same hscode and description already exists"}

    # Generate a new unique ID for the QK record (you can use a timestamp or a UUID)
    new_id = str(uuid.uuid4())  # Import 'uuid' library for UUID generation

    # Create a new QK record with the generated ID
    new_qk = QK(id=new_id, hscode=item.hscode, description=item.description, time=datetime.now(), own=item.own)

    # Convert the new QK record to a Pandas DataFrame row
    new_row = pd.DataFrame([new_qk.dict()])

    # Append the new row to the existing DataFrame
    df = pd.concat([df, new_row], ignore_index=True)

    # Save the updated DataFrame back to the CSV file
    df.to_csv("data/qk.csv", index=False)

    return {"message": "QK record added successfully", "id": new_id}



@app.delete("/removeqk/{id}")
async def remove_qk(id: str):
    # Load the existing data from the CSV file
    df = pd.read_csv("data/qk.csv")

    # Check if the QK record with the specified ID exists
    if id in df['id'].values:
        # Remove the row with the specified ID
        df = df[df['id'] != id]

        # Save the updated DataFrame back to the CSV file
        df.to_csv("data/qk.csv", index=False)

        return {"message": f"QK record with ID {id} removed successfully"}
    else:
        return {"message": f"QK record with ID {id} not found"}

@app.get("/updateqk")
async def update_qk(id: str, description: str):
    # Load the existing data from the CSV file
    df = pd.read_csv("data/qk.csv")

    # Check if the QK record with the specified ID exists
    if id in df['id'].values:
        # Update the 'description' for the row with the specified ID
        df.loc[df['id'] == id, 'description'] = description

        # Save the updated DataFrame back to the CSV file
        df.to_csv("data/qk.csv", index=False)

        return {"message": f"QK record with ID {id} updated successfully"}
    else:
        return {"message": f"QK record with ID {id} not found"}

if __name__ == '__main__':
    uvicorn.run("search_api:app", host="localhost", port=2211, reload=True)