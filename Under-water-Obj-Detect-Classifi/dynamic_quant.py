from __future__ import division

from DG_model import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import os
import torch.onnx 

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
# def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size,augment=True):
#     model.eval()

#     # Get dataloader
#     dataset = ListDataset(path, img_size=img_size, augment=augment, multiscale=False)
#     dataloader = torch.utils.data.DataLoader(
#         dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
#     )

#     Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

#     labels = []
#     sample_metrics = []  # List of tuples (TP, confs, pred)
#     for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
#         # imgshow(targets[targets[:,0]<0.1], imgs[0,:,:,:], 'out.jpg')
#         # Extract labels
#         labels += targets[:, 1].tolist()
#         # Rescale target
#         targets[:, 2:] = xywh2xyxy(targets[:, 2:])
#         targets[:, 2:] *= img_size

#         imgs = Variable(imgs.type(Tensor), requires_grad=False)

#         with torch.no_grad():
#             print(imgs.size())
#             outputs = model(imgs)
#             outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
#             # print(outputs.size())
#         sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

#     # Concatenate sample statistics
#     if len(sample_metrics) == 0:
#         precision, recall, AP, f1, ap_class = 0,0,0,0,0
#     else:
#         true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
#         precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

#     return precision, recall, AP, f1, ap_class



#Function to Convert to ONNX 
def Convert_ONNX(): 

    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor  
    dummy_input = torch.randn(1,3,224,224, requires_grad=True)  

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "ImageClassifier.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=11,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}}
         ) 
    print(" ") 
    print('Model has been converted to ONNX')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="./checkpoints/yolov3_ckpt_120.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco_marine.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--augment", type=bool, default=False, help="test in type8 dataset")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # print(device)
    device  = "cpu"

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        c = torch.load(opt.weights_path)
        # print(c.keys())
        # model.load_state_dict(torch.load(opt.weights_path)['model'])
        model.load_state_dict(torch.load(opt.weights_path))
        
    print(model)
    
    inp = torch.rand(1,3,224,224,requires_grad=True)
    
    op = model(inp)
    
    print(len(op))
        
    # Convert_ONNX() 
    
    # quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Conv2d, torch.nn.Linear}, dtype=torch.qint8) 
    quantized_model = torch.quantization.quantize_dynamic(model, dtype=torch.quint8) 
    
    torch.save(quantized_model,'quant.pt')
        

    # print("Compute mAP...")

    # precision, recall, AP, f1, ap_class = evaluate(
    #     model,
    #     path=valid_path,
    #     iou_thres=opt.iou_thres,
    #     conf_thres=opt.conf_thres,
    #     nms_thres=opt.nms_thres,
    #     img_size=opt.img_size,
    #     batch_size=8,
    #     augment = opt.augment
    # )

    # print("Average Precisions:")
    # for i, c in enumerate(ap_class):
    #     print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    # print(f"mAP: {AP.mean()}")
