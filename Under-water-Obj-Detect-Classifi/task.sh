# python DG_train.py --pretrained_weights ./weights/darknet53.conv.74 --batch_size 8
python test.py --weights_path ./weights/DGyolov3.pth --batch_size 32 --augment True
# python dynamic_quant.py --weights_path ./weights/DGyolov3.pth --batch_size 32 --augment True