from train_detector import Detector

img_dir = './'
root_dir = 'root_dir'
coco_dir = 'coco_dir'
set_dir = 'Images'

gtf = Detector()
gtf.Train_Dataset(root_dir, coco_dir, img_dir, set_dir, batch_size=16, use_gpu=False)
gtf.Model(model_name="resnet50")
gtf.Set_Hyperparams(lr=0.0001, val_interval=1, print_interval=20)
gtf.Train(num_epochs=2, output_model_name="final_model.pt")