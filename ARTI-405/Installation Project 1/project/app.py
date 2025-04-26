from fastapi import FastAPI, File, UploadFile, HTTPException
from inference import predict
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = FastAPI()

tmp_dir = '/tmp'
if not os.path.isdir(tmp_dir):
    os.makedirs(tmp_dir)

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Save upload to temporary file
    try:
        file_path = os.path.join(tmp_dir, file.filename)
        contents = await file.read()
        with open(file_path, 'wb') as f:
            f.write(contents)
        # Run prediction
        result = predict(file_path)
        return {"filename": file.filename, "prediction": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))