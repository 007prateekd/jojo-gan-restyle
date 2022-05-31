from torch import optim
from torch.nn import functional as F
from tqdm import tqdm
from pretrained_style import *
from time import time


def align_style_images(style):
    '''Returns the styled images after being aligned and stores them.'''

    style_path = os.path.join("style_images", style)
    assert os.path.exists(style_path), f"{style_path} does not exist!"
    style_aligned_path = os.path.join("style_images_aligned", f"{strip_path_extension(style)}.png")
    if not os.path.exists(style_aligned_path):
        style_aligned = align_face(style_path)
        style_aligned.save(style_aligned_path)
    else:
        style_aligned = Image.open(style_aligned_path).convert("RGB")
    return style_aligned


def get_style_code(style, style_aligned):
    '''Returns the style code for the given style image by applying e4e.'''
    
    style_code_path = os.path.join("inversion_codes", f"{style}.pt")
    if not os.path.exists(style_code_path):
        latent = e4e_projection(style_aligned, style_code_path, device)
    else:
        latent = torch.load(style_code_path)["latent"]
    return latent


def preprocess_styles(styles, transform):
    '''
    Returns the modified style images, aligned and transformed,
    along with the style codes for the style images.
    '''
    
    targets, latents = [], []
    for style in styles:
        style_aligned = align_style_images(style)
        style = strip_path_extension(style)
        latent = get_style_code(style, style_aligned)
        targets.append(transform(style_aligned).to(device))
        latents.append(latent.to(device))
    targets = torch.stack(targets, 0)
    latents = torch.stack(latents, 0)
    return targets, latents


def load_original_discriminator():
    '''Loads the pretrained StyleGAN2 discriminator.'''

    discriminator = Discriminator(1024, 2).eval().to(device)
    ckpt = torch.load("models/stylegan2-ffhq-config-f.pt", map_location=lambda storage, loc: storage)
    discriminator.load_state_dict(ckpt["d"], strict=False)
    return discriminator


def load_original_generator(latent_dim):
    '''Loads the pretrained StyleGAN2 generator with the given latent dimension.'''

    original_generator = Generator(1024, latent_dim, 8, 2).to(device)
    ckpt = torch.load("models/stylegan2-ffhq-config-f.pt", map_location=lambda storage, loc: storage)
    original_generator.load_state_dict(ckpt["g_ema"], strict=False)
    return original_generator


def finetune_generator(styles, transform, alpha, preserve_color, num_iter):
    '''
    Fine-tunes the original StyleGAN2 generator by minimizing the L1 loss between
    the discriminator activations    for the reference images and the generated images.
    '''
    
    targets, latents = preprocess_styles(styles, transform)
    discriminator = load_original_discriminator()
    generator = load_original_generator(latent_dim=512)
    g_optim = optim.Adam(generator.parameters(), lr=2e-3, betas=(0, 0.99))
    alpha = 1 - alpha
    id_swap = [9, 11, 15, 16, 17] if preserve_color else list(range(7, generator.n_latent))
    for _ in tqdm(range(num_iter)):
        mean_w = generator.get_latent(
            torch.randn([latents.size(0), latent_dim]).to(device)
        ).unsqueeze(1).repeat(1, generator.n_latent, 1)
        in_latent = latents.clone()
        in_latent[:, id_swap] = alpha * latents[:, id_swap] + (1 - alpha) * mean_w[:, id_swap]
        img = generator(in_latent, input_is_latent=True)
        with torch.no_grad():
            real_feat = discriminator(targets)
        fake_feat = discriminator(img)
        loss = sum([F.l1_loss(a, b) for a, b in zip(fake_feat, real_feat)]) / len(fake_feat)
        g_optim.zero_grad()
        loss.backward()
        g_optim.step()
    return generator


def latent_to_image(file_name, projection):
    '''Converts the latent vector to an image by passing it through the StyleGAN2 generator.'''
    
    aligned_face, name = align_face_helper(file_name)
    my_w = projection(aligned_face, name, device).unsqueeze(0)   
    original_generator = load_original_generator(latent_dim=512)
    with torch.no_grad():
        original_generator.eval()
        my_sample = original_generator(my_w, input_is_latent=True)
    display_image(utils.make_grid(my_sample, normalize=True, value_range=(-1, 1)), title="Generated Image")


def main(file_name, styles):
    '''
    Aligns the target image. Then fine-tunes the original generator for the given style(s). 
    The target image's code is then passed through this generator.
    Finally, the stylized image is transformed and displayed along with the reference and target image.
    '''
    
    aligned_face, name = align_face_helper(file_name)
    transform = get_transform(1024, 0.5, 0.5)
    generator = finetune_generator(styles, transform, alpha=1.0, preserve_color=False, num_iter=300)
    my_sample = generate_sample(aligned_face, name, 3000, generator)
    face = transform(aligned_face).unsqueeze(0).to(device)
    style_images = transform_style_images(styles, transform)
    display_image(utils.make_grid(style_images, normalize=True, value_range=(-1, 1)), title="References")
    my_output = torch.cat([face, my_sample], 0)
    display_image(utils.make_grid(my_output, normalize=True, value_range=(-1, 1)), title="My Sample")


if __name__ == "__main__":
    file_name = "Photo.jpeg"
    # styles = ["sketch1.jpeg", "sketch2.jpeg", "sketch3.jpeg", "sketch4.jpeg"] 
    # main(file_name, styles)
    start = time()
    latent_to_image(file_name, e4e)
    end = time()
    print(f"Time taken: {end - start:0.2f}s")