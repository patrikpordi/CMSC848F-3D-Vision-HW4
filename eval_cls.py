import numpy as np
import argparse

import torch
from data_loader import get_data_loader
from models import cls_model
from utils import create_dir, visualize_classification_result, rotate_x, rotate_y, rotate_z

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Rotation arguments
    
    parser.add_argument('--Rotation_XYZ', type=float, nargs=3, default=None, help='Amount of rotation around x, y, and z axes')
    


    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='model_epoch_0')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")
    parser.add_argument('--main_dir', type=str, default='./data/')
    parser.add_argument('--task', type=str, default="cls", help='The task: cls or seg')
    parser.add_argument('--batch_size', type=int, default=32, help='The number of images in a batch.')
    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')
    parser.add_argument('--num_workers', type=int, default=12, help='The number of threads to use for the DataLoader.')

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')
    
    
    return parser

def rotate(data_b, args):
    rotated_data = data_b.clone()

    x_angle, y_angle, z_angle = args.Rotation_XYZ
    rotated_data = rotate_x(rotated_data, x_angle, args)
    rotated_data = rotate_y(rotated_data, y_angle, args)
    rotated_data = rotate_z(rotated_data, z_angle, args)

    return rotated_data



if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    directory = args.output_dir+'/cls'
    create_dir(directory)
    directory = directory + '/'
    # print(args.num_points)
    if args.num_points != 10000:
        directory = directory+f'num_points_{args.num_points}'
        create_dir(directory)
        directory = directory + '/'
        num_points = args.num_points
    else:
        num_points = 10000
    rotation = args.Rotation_XYZ

    dataloader_cls = get_data_loader(args=args, train=False)
    # ------ TO DO: Initialize Model for Classification Task ------
    # model = 
    model = cls_model()
    model.to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/cls/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))
    labels = {0: 'chair', 1: 'vases', 2: 'lamps'}
    correct = 0
    num_obj = 0

    if rotation:
        print("Rotated data")
    else:
        print("Normal data")

    for i, batch in enumerate(dataloader_cls):
        data_b, data_b_label = batch
        data_b = data_b.to(args.device)
        data_b_label = data_b_label.to(args.device)

        selected_indices = torch.randint(0, data_b.shape[1], (args.num_points,))
        data_b = data_b[:, selected_indices, :]

        

        if rotation:
            rotated_data = rotate(data_b, args)
            
            output_rot_dir = f'Rotation_XYZ_{args.Rotation_XYZ[0]}_{args.Rotation_XYZ[1]}_{args.Rotation_XYZ[2]}/'
            rotated_data = rotate(data_b, args)
            create_dir(directory + output_rot_dir)

            # Make Prediction
            with torch.no_grad():
                prediction_label = model(rotated_data.to(args.device))
                prediction_label = prediction_label.max(dim=1)[1]

            prediction = [labels[label.item()] for label in prediction_label]
            GT = [labels[label.item()] for label in data_b_label]

            correct += prediction_label.eq(data_b_label.data).cpu().sum().item()
            num_obj += data_b_label.size()[0]
        else:
            # Make Prediction for non-rotated data
            with torch.no_grad():
                prediction_label = model(data_b.to(args.device))
                prediction_label = prediction_label.max(dim=1)[1]

            prediction = [labels[label.item()] for label in prediction_label]
            GT = [labels[label.item()] for label in data_b_label]

            correct += prediction_label.eq(data_b_label.data).cpu().sum().item()
            num_obj += data_b_label.size()[0]

        for idx in range(data_b.shape[0]):
            # Determine the class-specific paths
            if args.Rotation_XYZ:
                if GT[idx] == prediction[idx]:
                    prediction_path = output_rot_dir + GT[idx] + '/'
                else:
                    prediction_path = output_rot_dir + GT[idx] + '/fail/'
            else:
                if GT[idx] == prediction[idx]:
                    prediction_path = GT[idx] + '/'
                else:
                    prediction_path = GT[idx] + '/fail/'

            new_output_dir = directory + prediction_path
            create_dir(new_output_dir)

            if args.Rotation_XYZ:
                # Visualize the classification
                visualize_classification_result(rotated_data[idx], data_b_label[idx],
                            new_output_dir + f'Cls{idx}GT_{GT[idx]}_Pred_{prediction[idx]}.gif', args.device,args)
            else:
                # Visualize the classification
                visualize_classification_result(data_b[idx], data_b_label[idx],
                            new_output_dir + f'Cls{idx}GT_{GT[idx]}_Pred_{prediction[idx]}.gif', args.device,args)


    
    test_accuracy = correct / num_obj
    print ("Test accuracy: {}".format(test_accuracy))

