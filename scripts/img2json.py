allset = True
print("Starting module 'img2json' with Imports")
try:
    import os
    import json
    from collections import defaultdict
except:
    print("os or json or collections import failed.")
    allset = False
try:
    import torch
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
except:
    print("torch installation broken or not found.")
    allset = False
try:
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
except:
    print("transformers or BitsAndBytes installation broken or not found.")
    allset = False
try:
    from PIL import Image
except:
    print("PIL installation not found")
    allset = False

if(allset == False):
    raise ImportError("Fix the existing issues.")

print("Imports success. Loading functions.")

# constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=10, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=10):
    image = Image.open(image_file).convert('RGB')
    # image = image_file # pil formatted already
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

imglist = os.listdir("tmp")
all_images = defaultdict(list)

for i in imglist:
    toks = i.split("_][_")
    all_images[toks[1]].append(f"tmp/{i}")

print("Loading models in memory")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModel.from_pretrained("OpenGVLab/InternVL3-2B", torch_dtype=torch.float16, attn_implementation="flash_attention_2", low_cpu_mem_usage = True, trust_remote_code=True, device_map=device,quantization_config=quantization_config).eval()
tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL3-2B",device_map=device)

text_outputs = {}  # {doc_id: [page_text1, page_text2, ...]}
generation_config = dict(max_new_tokens=2048, do_sample=True)

for doc_id in all_images.keys():
    page_texts = []
    response_output = []
    print("Performing analysis for Document:",doc_id)
    for img in all_images[doc_id]:
        with torch.no_grad():
            pixel_values = load_image(img, max_num=10).to(torch.float16).cuda()
            question = """<image>
            You are an Image Descriptor. Your task is to analyze the given image.
            You must write out the text given in this image. You must describe the images if they are of any relevence or importance.
            Sometimes the contexts might be interrelated to the history. End with 2 escape newline sequence."""
            response, page_texts = model.chat(tokenizer, pixel_values, question, generation_config, history=page_texts, return_history=True)
            response_output.append(response)
        torch.cuda.empty_cache()
    text_outputs[doc_id] = response_output
    print("Document analysis completed, dumping data...")
    with open(f"tmp/text_outputs_{doc_id}.json", "w", encoding="utf-8") as jf:
        json.dump(text_outputs, jf, indent=4)
    text_outputs={}


if len(os.listdir("tmp"))>0:
    print("Cleaning tempdir...")
    for i in os.listdir("tmp"):
        if i.endswith(".png"):
            os.remove(f"tmp/{i}")

print("Cleaning Tempdir completed. Only JSON Files remain.")
print("Module 'img2json' completed. Pleae wait while next function loads.")