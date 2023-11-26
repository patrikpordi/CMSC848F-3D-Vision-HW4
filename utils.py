import os
import torch
import pytorch3d
from pytorch3d.renderer import (
    AlphaCompositor,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
)
import imageio
import numpy as np

def save_checkpoint(epoch, model, args, best=False):
    if best:
        path = os.path.join(args.checkpoint_dir, 'best_model.pt')
    else:
        path = os.path.join(args.checkpoint_dir, 'model_epoch_{}.pt'.format(epoch))
    torch.save(model.state_dict(), path)
    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)

def create_dir(directory):
    """
    Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_points_renderer(
    image_size=256, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer


def viz_seg (verts, labels, path, device,args):
    """
    visualize segmentation result
    output: a 360-degree gif
    """
    image_size=256
    background_color=(1, 1, 1)
    colors = [[1.0,1.0,1.0], [1.0,0.0,1.0], [0.0,1.0,1.0],[1.0,1.0,0.0],[0.0,0.0,1.0], [1.0,0.0,0.0]]

    # Construct various camera viewpoints
    dist = 3
    elev = 0
    azim = [180 - 12*i for i in range(30)]
    R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
    c = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)

    sample_verts = verts.unsqueeze(0).repeat(30,1,1).to(torch.float)
    sample_labels = labels.unsqueeze(0)
    sample_colors = torch.zeros((1,args.num_points,3))

    # Colorize points based on segmentation labels
    for i in range(6):
        sample_colors[sample_labels==i] = torch.tensor(colors[i])

    sample_colors = sample_colors.repeat(30,1,1).to(torch.float).to(device)

    point_cloud = pytorch3d.structures.Pointclouds(points=sample_verts, features=sample_colors).to(device)

    renderer = get_points_renderer(image_size=image_size, background_color=background_color, device=device)
    rend = renderer(point_cloud, cameras=c).cpu().numpy() # (30, 256, 256, 3)

    # Convert to uint8 and normalize to [0, 255]
    rend = (rend * 255).astype(np.uint8)
    imageio.mimsave(path, rend, fps=15,loop=0)

def visualize_classification_result(verts, labels, output_path, device, args):
    """
    Visualize point cloud classification result as a 360-degree gif.
    """
    image_size = 256
    background_color = (1, 1, 1)
    colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    # Construct various camera viewpoints
    dist = 3
    elev = 0
    azim = [180 - 12 * i for i in range(30)]
    R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
    c = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)

    sample_verts = verts.unsqueeze(0).repeat(30, 1, 1).to(torch.float)
    sample_labels = labels.unsqueeze(0)
    sample_colors = torch.zeros((1, args.num_points, 3))
    sample_colors[0] = torch.tensor(colors[labels])
    sample_colors = sample_colors.repeat(sample_verts.shape[0], 1, 1).to(torch.float).to(device)

    point_cloud = pytorch3d.structures.Pointclouds(points=sample_verts, features=sample_colors).to(device)

    renderer = get_points_renderer(image_size=image_size, background_color=background_color, device=device)
    rend = renderer(point_cloud, cameras=c).cpu().numpy()  # (30, 256, 256, 3)
    rend = (255 * rend).astype(np.uint8)

    imageio.mimsave(output_path, rend, fps=15, loop=0)

def rotate_x(input_data, angle, args):
    """
    Rotate the input data around the x-axis.

    Parameters:
    - input_data: torch.Tensor or numpy array, input data to be rotated
    - angle: float, rotation angle in degrees

    Returns:
    - rotated_data: torch.Tensor, rotated input data
    """
    if not isinstance(input_data, torch.Tensor):
        input_data = torch.tensor(input_data).to(args.device)

    angle_rad = torch.deg2rad(torch.tensor(angle, dtype=torch.float32)).to(args.device)
    rotation_matrix = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(angle_rad), -torch.sin(angle_rad)],
        [0, torch.sin(angle_rad), torch.cos(angle_rad)]
    ], dtype=torch.float32).to(args.device)

    rotated_data = torch.matmul(input_data, rotation_matrix)
    return rotated_data

def rotate_y(input_data, angle, args):
    """
    Rotate the input data around the y-axis.

    Parameters:
    - input_data: torch.Tensor or numpy array, input data to be rotated
    - angle: float, rotation angle in degrees

    Returns:
    - rotated_data: torch.Tensor, rotated input data
    """
    if not isinstance(input_data, torch.Tensor):
        input_data = torch.tensor(input_data).to(args.device)

    angle_rad = torch.deg2rad(torch.tensor(angle, dtype=torch.float32)).to(args.device)
    rotation_matrix = torch.tensor([
        [torch.cos(angle_rad), 0, torch.sin(angle_rad)],
        [0, 1, 0],
        [-torch.sin(angle_rad), 0, torch.cos(angle_rad)]
    ], dtype=torch.float32).to(args.device)

    rotated_data = torch.matmul(input_data, rotation_matrix)
    return rotated_data

def rotate_z(input_data, angle, args):
    """
    Rotate the input data around the z-axis.

    Parameters:
    - input_data: torch.Tensor or numpy array, input data to be rotated
    - angle: float, rotation angle in degrees

    Returns:
    - rotated_data: torch.Tensor, rotated input data
    """
    if not isinstance(input_data, torch.Tensor):
        input_data = torch.tensor(input_data).to(args.device)

    angle_rad = torch.deg2rad(torch.tensor(angle, dtype=torch.float32)).to(args.device)
    rotation_matrix = torch.tensor([
        [torch.cos(angle_rad), -torch.sin(angle_rad), 0],
        [torch.sin(angle_rad), torch.cos(angle_rad), 0],
        [0, 0, 1]
    ], dtype=torch.float32).to(args.device)

    rotated_data = torch.matmul(input_data, rotation_matrix)
    return rotated_data