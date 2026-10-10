import sys
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

"""
Starter code for Test Time Augmentation
This will automatically download the dataset, CLIP model, run preprocessing and inference and evaluation.
Your task is to update GDA specific functions.

****Update the functions spanning lines 222-268 ****

You may need to install torch, PIL, tqdm, transformers, datasets using pip

"""



import os
from datasets import load_dataset
from PIL import Image
import glob

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
import time


### Code to download the images
################################################################
###Choose where to save the files locally
SAVE_DIR = 'data/tta'
os.makedirs(SAVE_DIR, exist_ok=True)

existing_images_count = len(glob.glob(os.path.join(SAVE_DIR, "**", "*.jpg"), recursive=True))
if existing_images_count == 50000:
    print("Skipping the download. Dataset alredy exists")
    class_names = []
    with open("imagenet_classes.txt", "r") as f:
        for line in f:
            _, name = line.strip().split(": ", 1)
            class_names.append(name)
else:
    print("Downloading and saving 50,000 validation images...")
    dataset = load_dataset(
        "parquet", 
        data_files={"val": "hf://datasets/evanarlian/imagenet_1k_resized_256/data/val-*.parquet"}, 
        split="val"
    )
    class_names = dataset.features["label"].names
    
    # Fixed indentation here
    with open("./imagenet_classes.txt", "w") as f:
        for idx, name in enumerate(class_names):
            f.write(f"{idx}: {name}\n")
    print("Class names successfully saved to imagenet_classes.txt")

    for idx, sample in enumerate(dataset):
        image = sample["image"]
        label = sample["label"]  # Integer 0-999
        
        # Create a subfolder for each class to keep it organized
        class_dir = os.path.join(SAVE_DIR, f"class_{label}")
        os.makedirs(class_dir, exist_ok=True)
        
        # Save the image file
        image_path = os.path.join(class_dir, f"img_{idx}.jpg")
        image.save(image_path)

    print(f"Done! All 50,000 images saved to: {SAVE_DIR}")


class_names = []
with open("imagenet_classes.txt", "r") as f:
    for line in f:
        _, name = line.strip().split(": ", 1)
        class_names.append(name)

###Clip will use these prompts for generating text embeddings
prompted_labels = [f"a photo of a {name}." for name in class_names]
##########################################################





device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    device = 'mps'
print(f"Using device: {device}")


################# Dataset and Dataloader #########################

class LocalImageNetDataset(Dataset):
    def __init__(self, base_dir):
        self.image_paths = glob.glob(os.path.join(base_dir, "**", "*.jpg"), recursive=True)
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        # Parse the ground-truth integer ID out of the folder structure "class_XYZ"
        folder_name = os.path.basename(os.path.dirname(img_path))
        label_id = int(folder_name.replace("class_", ""))
        
        return image, label_id


# SAVE_DIR = "./local_imagenet_val"
dataset = LocalImageNetDataset(SAVE_DIR)


def collate_fn(batch):
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return images, labels


data_loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)


############################## Load model #######################################

MODEL_ID = "openai/clip-vit-base-patch32"
LOCAL_MODEL_DIR = "code/q2/clip"

if not os.path.exists(LOCAL_MODEL_DIR):
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    print(f"Downloading and saving {MODEL_ID} locally to {LOCAL_MODEL_DIR}...")
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model = CLIPModel.from_pretrained(MODEL_ID)
    processor.save_pretrained(LOCAL_MODEL_DIR)
    model.save_pretrained(LOCAL_MODEL_DIR)

else:
    ###Load the models if already downloaded
    processor = CLIPProcessor.from_pretrained(LOCAL_MODEL_DIR)
    model = CLIPModel.from_pretrained(LOCAL_MODEL_DIR).to(device)
    model.eval()

model=model.to(device)


print("Pre-computing text embeddings for 1,000 classes...")
with torch.no_grad():
    text_inputs = processor(text=prompted_labels, padding=True, return_tensors="pt").to(device)
    text_features = model.get_text_features(**text_inputs)
    # Normalize features to unit length for cosine similarity
    text_features = text_features.pooler_output if hasattr(text_features,'pooler_output') else text_features
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
# assert False, text_features.shape

################## Uncomment to run zero-shot evaluation ###############

top1_correct = 0
top5_correct = 0
total_images = 0

print("Starting Zero-Shot Evaluation...")
with torch.no_grad():
    for images, labels in tqdm(data_loader, desc="Evaluating Batches"):
        labels = labels.to(device)
         
        # Process image batch
        image_inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        image_features = model.get_image_features(**image_inputs)
        image_features = image_features.pooler_output if hasattr(image_features,'pooler_output') else image_features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Compute Cosine Similarity Matrix (Batch_Size x 1000)
        # Similarity ranges from -1 to 1; scales logits appropriately
        similarity = (image_features @ text_features.T) * model.logit_scale.exp()

        
        # Calculate Accuracies
        total_images += labels.size(0)
        
        # Top-1 accuracy check
        _, top1_preds = similarity.topk(1, dim=-1)

        
        top1_correct += (top1_preds.squeeze(-1) == labels).sum().item()
        
        # Top-5 accuracy check
        _, top5_preds = similarity.topk(5, dim=-1)
        top5_correct += (top5_preds == labels.unsqueeze(-1)).any(dim=-1).sum().item()


print("\n=== Evaluation Results ===")
print(f"Total Evaluated: {total_images} / 50000")
print(f"Top-1 Accuracy:  {(top1_correct / total_images) * 100:.2f}%")
print(f"Top-5 Accuracy:  {(top5_correct / total_images) * 100:.2f}%")

################################################################################

class OnlineLabelEstimator:
    def __init__(self, num_classes=1000, feature_dim=512,
                 bank_capacity=16, beta=1.0, eps=1e-4):
        self.K = num_classes
        self.d = feature_dim
        self.L = bank_capacity
        self.beta = beta
        self.eps  = eps          

        # Storage
        self.prior_means  = torch.zeros(self.K, self.d)
        self.bank_features = torch.zeros(self.K, self.L, self.d)   # [K, L, d]
        self.bank_clip_y   = torch.zeros(self.K, self.L)           # [K, L]
        self.bank_counts   = torch.zeros(self.K, dtype=torch.long) # [K]

        
        arange = torch.arange(self.L)
        
        self._arange_L = arange  

    def to(self, device):
        self.prior_means   = self.prior_means.to(device)
        self.bank_features = self.bank_features.to(device)
        self.bank_clip_y   = self.bank_clip_y.to(device)
        self.bank_counts   = self.bank_counts.to(device)
        self._arange_L     = self._arange_L.to(device)
        return self

    
    def _mask(self):
        """Boolean mask [K, L] : True where slot is occupied."""
        return self._arange_L.unsqueeze(0) < self.bank_counts.unsqueeze(1) # [k,l]

    ########### Write your code for computing mean and covariance here###############33
    def compute_optimal_means_and_covariance(self):
        # assert False, text_features.shape
        ### *****
        #!! this is presumably called after warm-up
        mask = self._mask()
        s = (self.bank_clip_y*mask).sum(1).clamp_min(1e-8)
        alpha = s/(s+self.beta) #/ (k,) shape
        mu_prime = (self.bank_features*mask.unsqueeze(-1)*self.bank_clip_y.unsqueeze(-1)).sum(1) \
                     / s.unsqueeze(-1)
        mu_star = alpha.unsqueeze(-1)*mu_prime + (1-alpha).unsqueeze(-1)*self.prior_means #/ (k,d)
        Nb = self.bank_counts.sum().clamp_min(1)
        centered = self.bank_features - mu_star.unsqueeze(1) #/ (k,l,d)
        centermasked = centered*mask.unsqueeze(-1)
        sigma = torch.einsum('kdl,kle->de',centermasked.transpose(2,1),centermasked) / Nb #/ (d,d)
        sigma_inv = self.d*torch.linalg.inv((Nb-1)*sigma+torch.trace(sigma)*torch.eye(self.d).to(device))
        ### *****
 
        return mu_star, sigma_inv


    def compute_gda_predictions(self, mu_star, sigma_inv, x_i):
        ####Write code to compute w and b below:

        ### *****
        w = mu_star@sigma_inv #->soft predictions [k,d]
        b = -0.5*(w*mu_star).sum(-1)
        ### *****

        return x_i @ w.T + b.unsqueeze(0)        # [B, K]


    def estimate_labels(self, hat_y, mu_star, sigma_inv, x_i, gda_scale=1.0):
        # assert False, x_i.shape
        y_tilde = self.compute_gda_predictions(mu_star, sigma_inv, x_i)  # [B, K]

        mask     = self._mask()
        # centered = self.bank_features - mu_star.unsqueeze(1)             # [K, L, d]
        # S = (centered * mask.unsqueeze(-1)
        #            * self.bank_clip_y.unsqueeze(-1)).sum(1)             # [K, d]
        # A = x_i @ sigma_inv                                              # [B, d]
        # consistency = A @ S.T                                            # [B, K]

        ######Write code to compute z_star below:

        ### *****
        wij = torch.einsum('bd,kld->bkl',x_i,self.bank_features)
        wij = wij*mask.unsqueeze(0)
        bank = (wij*self.bank_clip_y.unsqueeze(0)).sum(-1)
        z_logits = torch.log(hat_y.clamp_min(1e-8)) + gda_scale*y_tilde+bank
        z_star = torch.softmax(z_logits,dim=-1)
        ### *****

        return z_star
        
    @torch.no_grad()
    def update_knowledge_bank_with_gt(self, x_i, true_labels):
        """Store features under their ground-truth class (evaluation diagnostic)."""
        B = x_i.size(0)
        for idx in range(B):
            k       = true_labels[idx].item()
            feature = x_i[idx]
            count   = self.bank_counts[k].item()

            if count < self.L:
                self.bank_features[k, count] = feature
                self.bank_clip_y[k, count]   = 1.0
                self.bank_counts[k] += 1
            else:
                self.bank_features[k] = torch.roll(self.bank_features[k], -1, dims=1)
                self.bank_clip_y[k]   = torch.roll(self.bank_clip_y[k], -1, dims=1)
                self.bank_features[k, 0] = feature
                self.bank_clip_y[k, 0]   = 1.0


def run_gda_eval(data_loader, model, processor, text_features, device,
                 bank_capacity=64,
                 beta=10.0,
                 gda_scale=5.0,
                 recompute_interval=5,
                 min_bank_size=2000,
                 use_gt=True):
    """
    GDA Test-Time Adaptation evaluation.
    """
    num_classes = len(prompted_labels)
    feature_dim = model.config.projection_dim

    estimator = OnlineLabelEstimator(
        num_classes=num_classes,
        feature_dim=feature_dim,
        bank_capacity=bank_capacity,
        beta=beta,
    ).to(device)

    top1_correct   = 0
    top5_correct   = 0
    total_images   = 0
    gda_batches    = 0
    zero_batches   = 0

    # Initial GDA params (identity precision, zero means)
    estimator.prior_means = text_features #?.clone
    mu_star   = estimator.prior_means.clone()
    sigma_inv = torch.eye(feature_dim, device=device)

    print("\n" + "=" * 60)
    print("  GDA TEST-TIME ADAPTATION EVALUATION")
    print(f"  bank_capacity={bank_capacity}  beta={beta}  gda_scale={gda_scale}")
    print(f"  feature_dim={feature_dim}  num_classes={num_classes}")
    print(f"  min_bank_size={min_bank_size}  use_gt={use_gt}")
    print("=" * 60)

    batch_idx = 0
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="GDA-TTA"):
            labels = labels.to(device)
            batch_idx += 1

            # ── (1) CLIP image features ─────────────────────────────────────
            image_inputs = processor(images=images, return_tensors="pt",
                                    padding=True).to(device)
            image_features = model.get_image_features(**image_inputs)
            #?? hf>=5.0 does not output raw tensors anymore
            image_features = image_features.pooler_output if hasattr(image_features,'pooler_output') else image_features
            image_features = image_features / image_features.norm(
                                dim=-1, keepdim=True)

            # ── (2) CLIP zero-shot soft predictions ─────────────────────────
            raw_sim = (image_features @ text_features.T) * model.logit_scale.exp()
            hat_y   = F.softmax(raw_sim, dim=-1)                    # [B, K]

            # ── (3) Check if bank is warm enough for GDA ───────────────────
            bank_size = int(estimator.bank_counts.sum().item())

            if bank_size >= min_bank_size:
                # Recompute GDA params periodically
                if batch_idx % recompute_interval == 1:
                    mu_star, sigma_inv = estimator.compute_optimal_means_and_covariance()
                    torch.mps.synchronize()

                # GDA-fused posterior
                z_star = estimator.estimate_labels(
                    hat_y, mu_star, sigma_inv, image_features,
                    gda_scale=gda_scale
                )
                gda_batches += 1
            else:
                # Bank not warm yet → pure CLIP zero-shot
                z_star = hat_y
                zero_batches += 1

            
            total_images += labels.size(0)

            _, top1_preds = z_star.topk(1, dim=-1)
            top1_correct += (top1_preds.squeeze(-1) == labels).sum().item()

            _, top5_preds = z_star.topk(5, dim=-1)
            top5_correct += (top5_preds == labels.unsqueeze(-1)).any(dim=-1).sum().item()

           
            if use_gt:
                # Ground-truth labels (validates GDA math, no feedback loop)
                estimator.update_knowledge_bank_with_gt(image_features, labels)
            else:
                # Pseudo-labels (true test-time adaptation)
                estimator.update_knowledge_bank(image_features, z_star)

    top1_acc = (top1_correct / total_images) * 100
    top5_acc = (top5_correct / total_images) * 100

    print(f"\n  Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"  Top-5 Accuracy: {top5_acc:.2f}%")
    print(f"  Batches using GDA:       {gda_batches}/{batch_idx}")
    print(f"  Batches using zero-shot: {zero_batches}/{batch_idx}")
    print(f"  Total bank samples:      {int(estimator.bank_counts.sum().item())}")

    return top1_acc, top5_acc



########### RUN GDA #############################

t0 = time.time()
gda_top1, gda_top5 = run_gda_eval(
    data_loader, model, processor, text_features, device,
    bank_capacity=64,
    beta=1.0,
    gda_scale=5.0,            # ← start small (was 5.0)
    recompute_interval=5,
    min_bank_size=2000,
    use_gt=True,
)
t_gda = time.time() - t0

# --- Side-by-side report ---
print("\n" + "=" * 60)
print("  COMPARISON  (ImageNet-1k validation set)")
print("=" * 60)
print(f"  {'Method':<30} {'Top-1':>8} {'Top-5':>8} {'Time(s)':>8}")
print(f"  {'-'*56}")
print(f"  {'GDA Test-Time Adaptation':<30} {gda_top1:>7.2f}% {gda_top5:>7.2f}% {t_gda:>7.1f}")
print(f"  {'-'*56}")
print("=" * 60)