import torchvision.transforms as transforms
import torch
from angle_prediction.models import AngleModel
from PIL import Image
from angle_prediction.loss_function import code2angle
from math import degrees


# model part
def init_model_trans(ckpt_path=r'D:\wise_transportation\gitee_repo\mmyolo\ckpt\checkpoint-199-3_21.pth'):
    model = AngleModel().cuda()
    ckpt = torch.load(ckpt_path)
    ckpt = ckpt['model']
    missing, unexp = model.load_state_dict(ckpt)
    print(missing, unexp)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize(image.size[0] * 512 // image.size[1]),
        # transforms.Pad((100, 100)),
        # transforms.CenterCrop((512, 512)),
        transforms.Resize((512, 512)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return model, transform_test


def test(model, transform, x):
    x = transform(x).unsqueeze(0).to('cuda:0')
    model.eval()
    with torch.no_grad():
        target = torch.zeros(0).to('cuda:0')
        loss, pred, acc = model(x, target)
    return pred

def multi_image_test(model, transform, x):
    input = []
    for img in x:
        input.append(transform(img))
    x = torch.stack(input, dim=0).cuda()
    model.eval()
    with torch.no_grad():
        target = torch.zeros(x.shape[0]).to('cuda:0')
        loss, pred, acc = model(x, target)
    return pred


if __name__ == '__main__':
    # image part
    image_path = r'D:\wise_transportation\data\infer_4.jpg'
    # image_path = r'D:\Instant-NGP-for-RTX-3000-and-4000\angle_data\pic_dataset\1\2_265.jpg'
    image = Image.open(image_path).convert('RGB')
    model, trans = init_model_trans()
    output = test(model, trans, image)
    print(output)
