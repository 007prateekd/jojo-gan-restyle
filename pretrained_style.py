import torch
from torchvision import transforms, utils
from PIL import Image
from util import *
from model import *
from download_data import *
from e4e_projection import projection as e4e_projection


device = "cpu"
latent_dim = 512


def align_face_helper(file_name):
    '''
    Aligns the face by using the pre-downloaded facial
    landmarks model and returns the cropped face image.
    '''
    
    file_path = f"test_input/{file_name}"
    name = strip_path_extension(file_path) + ".pt"
    aligned_face = align_face(file_path)
    return aligned_face, name


def load_finetuned_generator(preserve_color, style):
    '''
    Loads the style-specific fine-tuned generator using stored weigths. 
    Can also preserve color of the target image if that particular checkpoint is stored.
    '''
    
    ckpt = f"{style}_preserve_color.pt" if preserve_color else f"{style}.pt"
    try:
        download_from_drive(ckpt)
    except:
        ckpt = f"{style}.pt"
        download_from_drive(ckpt)
    ckpt = torch.load(os.path.join("models", ckpt), map_location=lambda storage, loc: storage)
    generator = Generator(1024, latent_dim, 8, 2).to(device)
    generator.load_state_dict(ckpt["g"], strict=False)
    return generator


def generate_sample(aligned_face, name, seed, generator):
    '''
    Generates an image where the reference style is applied to the target image
    by passing the latent code of the target image through the pretrained generator.
    '''
    
    my_w = e4e_projection(aligned_face, name, device).unsqueeze(0)
    torch.manual_seed(seed)
    with torch.no_grad():
        generator.eval()
        my_sample = generator(my_w, input_is_latent=True)
    return my_sample


def get_transform(img_size, mean, std):
    '''Returns a transform to resize and normalize any image.'''

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((mean, mean, mean), (std, std, std)),
        ]
    )
    return transform


def transform_style_images(styles, transform):
    '''Returns an array of style images with the given transform applied.'''
    
    style_images = []
    dict = {"arcane_multi": "arcane_jinx", "sketch_multi": "sketch1"}
    for style in styles:
        style = dict[style] if style in dict.keys() else strip_path_extension(style)
        style_path = f"style_images_aligned/{style}.png"
        style_image = transform(Image.open(style_path))
        style_images.append(style_image)
    style_images = torch.stack(style_images, 0).to(device)
    return style_images


def main(file_name, style):
    '''
    Aligns the target image. Then loads the fine-tuned generator for 
    the given style and passes the image's code through it after which it is 
    transformed and displayed along with the reference and target image.
    '''
    
    style = [style]
    aligned_face, name = align_face_helper(file_name)
    generator = load_finetuned_generator(preserve_color=False, style=style[0])
    my_sample = generate_sample(aligned_face, name, 3000, generator)
    transform = get_transform(1024, 0.5, 0.5)
    face = transform(aligned_face).unsqueeze(0).to(device)
    style_image = transform_style_images(style, transform)
    my_output = torch.cat([style_image, face, my_sample], 0)
    display_image(utils.make_grid(my_output, normalize=True, value_range=(-1, 1)), title="Stylized Image")


if __name__ == "__main__":
    # options = ["art", "arcane_caitlyn", "arcane_jinx", "disney", "jojo", "jojo_yasuho", "sketch_multi"]
    style = "jojo"
    file_name = "Photo.jpeg"
    main(file_name, style)