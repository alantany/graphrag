from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from data_processor import process_data

app = FastAPI()

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # 处理上传的文件
    content = await file.read()
    result = process_data(content)
    return {"message": "文件上传成功", "result": result}

@app.get("/")
async def root():
    return {"message": "知识图谱生成系统API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)