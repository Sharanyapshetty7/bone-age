"""Bone age prediction interface (research demo, not for clinical use).

Needs stage2_512.pt in the same folder.
Set AGE_MEAN / AGE_STD to the values printed in your training notebook.
"""
import torch, torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as TF
import gradio as gr
from PIL import Image, ImageOps

CKPT = "stage2_512.pt"
AGE_MEAN, AGE_STD = 127.32226407613005,41.17428364568501  # <-- REPLACE with your printed training-split values
SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BoneAgeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        self.backbone.fc = nn.Identity()
        self.sex = nn.Sequential(nn.Linear(1, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2048 + 32, 256), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(256, 1))

    def forward(self, x, sex):
        return self.head(torch.cat([self.backbone(x), self.sex(sex)], 1)).squeeze(1)


model = BoneAgeNet().to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()


@torch.no_grad()
def predict(img, sex):
    if img is None:
        return "Please upload a hand X-ray."
    img = ImageOps.pad(img.convert("L"), (SIZE, SIZE), color=0)
    x = TF.to_tensor(img).repeat(3, 1, 1)
    x = TF.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0).to(DEVICE)
    s = torch.tensor([[1.0 if sex == "Male" else 0.0]], device=DEVICE)
    views = [x, torch.flip(x, [3]),
             TF.rotate(x, 5), TF.rotate(x, -5),
             TF.affine(x, 0, [10, 0], 1.0, 0), TF.affine(x, 0, [-10, 0], 1.0, 0)]
    p = torch.stack([model(v, s) for v in views])
    months = (p.mean().item() * AGE_STD) + AGE_MEAN
    spread = p.std().item() * AGE_STD          # disagreement between the 6 views
    yrs, mo = divmod(round(months), 12)
    return (f"### Estimated bone age: **{months:.0f} months** ({yrs} years {mo} months)\n"
            f"Typical error of this model: about ±6 months (88% of test cases within 12 months).\n\n"
            f"View-to-view spread: {spread:.1f} months "
            f"({'consistent' if spread < 3 else 'less certain, check the image quality'}).\n\n"
            f"*Research demo only. Not a medical device.*")


demo = gr.Interface(
    fn=predict,
    inputs=[gr.Image(type="pil", label="Hand/wrist X-ray (PA view)"),
            gr.Radio(["Female", "Male"], value="Female", label="Sex")],
    outputs=gr.Markdown(label="Result"),
    title="Bone Age Estimator",
    description="ResNet-50 trained on the RSNA Pediatric Bone Age dataset. Test MAE 6.2 months.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(share=True)