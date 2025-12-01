import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import net
from function import adaptive_instance_normalization, coral


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str, default=None,
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str, default='my_content',
                    help='Directory path to a batch of content images (default: my_content)')
parser.add_argument('--style', type=str, default=None,
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', type=str, default='my_style',
                    help='Directory path to a batch of style images (default: my_style)')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder.pth')

# Additional options
parser.add_argument('--content_size', type=int, default=512,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=512,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')

# Advanced options
parser.add_argument('--preserve_color', action='store_true',
                    help='If specified, preserve color of the content image')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')
parser.add_argument(
    '--style_interpolation_weights', type=str, default='',
    help='The weight for blending the style of multiple style images')

args = parser.parse_args()

do_interpolation = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

# Either --content or --contentDir should be given.
if not (args.content or args.content_dir):
    print("❌ Error: Content image parameter is required")
    print("📝 Usage:")
    print("   Method 1: python test.py --content <content_image_path> --style <style_image_path>")
    print("   Method 2: python test.py --content_dir <content_dir> --style_dir <style_dir>")
    print("   Method 3: python test.py  (using default paths: my_content/ and my_style/)")
    print("\n💡 Example:")
    print("   python test.py --content input/content/cornell.jpg --style input/style/woman_with_hat_matisse.jpg")
    exit(1)
if args.content:
    content_paths = [Path(args.content)]
else:
    content_dir = Path(args.content_dir)
    if not content_dir.exists():
        print(f"❌ Error: Content image directory not found: {content_dir}")
        print(f"💡 Please create directory {content_dir} and put your images in it, or use --content_dir to specify another path")
        exit(1)
    content_paths = [f for f in content_dir.glob('*') if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    if not content_paths:
        print(f"❌ Error: No image files found in directory {content_dir}")
        print(f"💡 Supported formats: .jpg, .jpeg, .png")
        exit(1)

# Either --style or --styleDir should be given.
if not (args.style or args.style_dir):
    print("❌ Error: Style image parameter is required")
    print("📝 Usage:")
    print("   python test.py --content <content_image> --style <style_image>")
    print("   or: python test.py --content_dir <content_dir> --style_dir <style_dir>")
    print("   or: python test.py  (using default paths: my_content/ and my_style/)")
    exit(1)
if args.style:
    style_paths = args.style.split(',')
    if len(style_paths) == 1:
        style_paths = [Path(args.style)]
    else:
        do_interpolation = True
        assert (args.style_interpolation_weights != ''), \
            'Please specify interpolation weights'
        weights = [int(i) for i in args.style_interpolation_weights.split(',')]
        interpolation_weights = [w / sum(weights) for w in weights]
else:
    style_dir = Path(args.style_dir)
    if not style_dir.exists():
        print(f"❌ Error: Style image directory not found: {style_dir}")
        print(f"💡 Please create directory {style_dir} and put style images in it, or use --style_dir to specify another path")
        exit(1)
    style_paths = [f for f in style_dir.glob('*') if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    if not style_paths:
        print(f"❌ Error: No image files found in directory {style_dir}")
        print(f"💡 Supported formats: .jpg, .jpeg, .png")
        exit(1)

decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

# Check if model files exist
if not Path(args.decoder).exists():
    print(f"❌ Error: Model file not found: {args.decoder}")
    print("📥 Please download model files from:")
    print("   https://github.com/naoto0804/pytorch-AdaIN/releases/tag/v0.0.0")
    print("   Required files:")
    print("   1. decoder.pth")
    print("   2. vgg_normalised.pth")
    print("   Please put them in the models/ directory")
    exit(1)

if not Path(args.vgg).exists():
    print(f"❌ Error: Model file not found: {args.vgg}")
    print("📥 Please download model files from:")
    print("   https://github.com/naoto0804/pytorch-AdaIN/releases/tag/v0.0.0")
    print("   Required files:")
    print("   1. decoder.pth")
    print("   2. vgg_normalised.pth")
    print("   Please put them in the models/ directory")
    exit(1)

decoder.load_state_dict(torch.load(args.decoder))
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

vgg.to(device)
decoder.to(device)

content_tf = test_transform(args.content_size, args.crop)
style_tf = test_transform(args.style_size, args.crop)

for content_path in content_paths:
    if do_interpolation:  # one content image, N style image
        style = torch.stack([style_tf(Image.open(str(p))) for p in style_paths])
        content = content_tf(Image.open(str(content_path))) \
            .unsqueeze(0).expand_as(style)
        style = style.to(device)
        content = content.to(device)
        with torch.no_grad():
            output = style_transfer(vgg, decoder, content, style,
                                    args.alpha, interpolation_weights)
        output = output.cpu()
        output_name = output_dir / '{:s}_interpolation{:s}'.format(
            content_path.stem, args.save_ext)
        save_image(output, str(output_name))

    else:  # process one content and one style
        for style_path in style_paths:
            content = content_tf(Image.open(str(content_path)))
            style = style_tf(Image.open(str(style_path)))
            if args.preserve_color:
                style = coral(style, content)
            style = style.to(device).unsqueeze(0)
            content = content.to(device).unsqueeze(0)
            with torch.no_grad():
                output = style_transfer(vgg, decoder, content, style,
                                        args.alpha)
            output = output.cpu()

            output_name = output_dir / '{:s}_stylized_{:s}{:s}'.format(
                content_path.stem, style_path.stem, args.save_ext)
            save_image(output, str(output_name))
            print(f"✅ Style transfer completed! Result saved to: {output_name}")
