from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(data= 'data_config.yaml',  epochs= 50, imgsz= 640, batch= 3, workers=1) # 用workers會當機
# results = model.val(data= 'data_config.yaml')

# CLI: yolo task=detect mode=train epochs=50 data=data_config.yaml model=yolov8n.pt imgsz=640 batch=3 (預設的batch是16!!!)
# CLI: yolo task=detect mode=predict model=best.pt source="0" show=True

# model = YOLO('best.pt')
# results = model.predict(source="data/predict", show= True, save= True)

# CLI: command line interface 