import argparse
import os

import cv2
import numpy as np
import torch

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from module.resnet18 import Resnet18


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default=r'H:\二尖瓣项目\dataset\whole_data\train\Barlow\Patient1\A2C\1.png',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
             'of cam_weights*activations')

    parser.add_argument(
        '--method',
        type=str,
        default='gradcam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == '__main__':
    """ python vit_gradcam.py -image-path <path_to_image>
    Example usage of using cam-methods on a VIT network.

    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    # load model weights

    # model_weight_path = "single_view_models/A2C/A2C_best_model_2_e50_lr=0.0001.pth"
    # model_weight_path = "single_view_models/A3C/A3C_best_model_2_e50_lr=0.0001.pth"
    # model_weight_path = "single_view_models/A4C/A4C_best_model_2_e50_lr=0.0001.pth"
    # model_weight_path ="3 CLASS Final/single_view_models/A2C/A2C_best_model_1_e50_lr=0.001.pth"
    # model_weight_path = "3 CLASS Final/single_view_models/A3C/A3C_best_model_1_e50_lr=0.001.pth"
    # model_weight_path ="3 CLASS Final/single_view_models/A4C/A4C_best_model_2_e50_lr=0.0001.pth"
    model_weight_path="3 CLASS Final/whole_model/best_model_2_e50_lr=0.0001.pth"
    model = torch.load(model_weight_path, map_location='cpu')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if args.use_cuda:
        model = model.cuda()

    target_layers = [model.model8[-1]]

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    if args.method == "ablationcam":
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   use_cuda=args.use_cuda,
                                   reshape_transform=reshape_transform,
                                   ablation_layer=AblationLayerVit())
    else:
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   use_cuda=args.use_cuda,
                                   )
    save_path = "Grad_3_class/A2C/whole/FED"
    os.makedirs(save_path, exist_ok=True)
    item_path = "dataset/final_3_class_dataset_single_view/A2C/train/FED"
    itemlist = os.listdir(item_path)
    for item in itemlist:
        item_name = os.path.join(item_path, item)
        save_name = os.path.join(save_path, item)
        rgb_img = cv2.imread(item_name)[:, :, ::-1]
        img_height, img_width = rgb_img.shape[0], rgb_img.shape[1]
        rgb_img = cv2.resize(rgb_img, (224, 224))
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        input_tensor = input_tensor.to(device)
        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested category.
        targets = None

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32

        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            eigen_smooth=args.eigen_smooth,
                            aug_smooth=args.aug_smooth)

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam)
        # cam_image = cv2.resize(cam_image, (img_width, img_height))

        cv2.imwrite(save_name, cam_image)
