from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import base64
from PIL import Image
import io
from util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img

app = FastAPI()

class InferRequest(BaseModel):
    prompt: str
    image_base64: str

yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt')
caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence")

def process(image_input, box_threshold, iou_threshold, use_paddleocr, imgsz):
    #TODO: Remove the need to save the image to disk
    image_save_path = 'imgs/saved_image_demo.png'
    image_input.save(image_save_path)
    image = Image.open(image_save_path)
    box_overlay_ratio = image.size[0] / 3200
    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }

    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image_save_path, display_img=False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold': 0.9}, use_paddleocr=use_paddleocr)
    text, ocr_bbox = ocr_bbox_rslt
    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image_save_path, yolo_model, BOX_TRESHOLD=box_threshold, output_coord_in_ratio=True, ocr_bbox=ocr_bbox, draw_bbox_config=draw_bbox_config, caption_model_processor=caption_model_processor, ocr_text=text, iou_threshold=iou_threshold, imgsz=imgsz)
    image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
    parsed_content_list = '\n'.join([f'icon {i}: ' + str(v) for i, v in enumerate(parsed_content_list)])
    return image, str(parsed_content_list)

@app.post("/infer")
async def infer(request: InferRequest):
    inputstr = request.prompt
    image_base64 = request.image_base64

    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
    except base64.binascii.Error:
        raise HTTPException(status_code=400, detail="Invalid base64 string")

    processed_image, parsed_content = process(image, box_threshold=0.05, iou_threshold=0.1, use_paddleocr=True, imgsz=640)

    buffered = io.BytesIO()
    processed_image.save(buffered, format="PNG")
    processed_image_base64 = base64.b64encode(buffered.getvalue()).decode()

    response = {
        'message': inputstr,
        'image': processed_image_base64,
        'parsed_content': parsed_content
    }

    return JSONResponse(content=response)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005, debug=True)