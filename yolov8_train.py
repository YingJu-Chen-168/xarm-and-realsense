from ultralytics import YOLO

# model = YOLO('yolov8n-seg.pt')
# results = model.train(data= 'data_config.yaml',  epochs= 100, imgsz= 640, batch= 8, workers=1) # 用workers會當機
# results = model.val(data= 'data_config.yaml')

# CLI: yolo task=segment mode=train epochs=100 data=data_config.yaml model=yolov8n-seg.pt imgsz=640 batch=8 (預設的batch是16!!!)
# CLI: yolo task=segment mode=predict model=best.pt source="data/predict/50.jpg" show=True

model = YOLO('yolov8n-seg.pt')
results = model.predict(source="data/predict", show= True, save= True)

# CLI: command line interface 