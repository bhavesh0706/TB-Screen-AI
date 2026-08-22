from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]

IMG_SIZE = 256

IMAGE_DIR = ROOT / "data" / "raw" / "LungSegmentation704" / "image"
MASK_DIR = ROOT / "data" / "raw" / "LungSegmentation704" / "mask"

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

model = tf.keras.models.load_model(
    ROOT / "weights" / "unet_best.keras",
    compile=False,
)

images = sorted(IMAGE_DIR.glob("*"))
masks = sorted(MASK_DIR.glob("*"))

_, val_imgs, _, val_masks = train_test_split(
    images,
    masks,
    test_size=0.2,
    random_state=42,
)

dice_scores = []
iou_scores = []

fig, axes = plt.subplots(5, 4, figsize=(12, 15))

for i in range(5):

    img = tf.keras.utils.load_img(val_imgs[i], target_size=(IMG_SIZE, IMG_SIZE))
    img_arr = tf.keras.utils.img_to_array(img)/255.0

    mask = tf.keras.utils.load_img(
        val_masks[i],
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
    )

    mask_arr = tf.keras.utils.img_to_array(mask)/255.0
    mask_bin = (mask_arr > 0.5).astype(np.float32)

    pred = model.predict(img_arr[None], verbose=0)[0]
    pred_bin = (pred > 0.5).astype(np.float32)

    inter = np.sum(mask_bin*pred_bin)
    union = np.sum(mask_bin)+np.sum(pred_bin)

    dice = (2*inter+1)/(union+1)
    iou = (inter+1)/(np.sum(mask_bin)+np.sum(pred_bin)-inter+1)

    dice_scores.append(dice)
    iou_scores.append(iou)

    axes[i,0].imshow(img_arr)
    axes[i,0].set_title("Original")
    axes[i,0].axis("off")

    axes[i,1].imshow(mask_bin.squeeze(), cmap="gray")
    axes[i,1].set_title("Ground Truth")
    axes[i,1].axis("off")

    axes[i,2].imshow(pred_bin.squeeze(), cmap="gray")
    axes[i,2].set_title("Prediction")
    axes[i,2].axis("off")

    axes[i,3].imshow(img_arr)
    axes[i,3].imshow(pred_bin.squeeze(), alpha=0.35, cmap="jet")
    axes[i,3].set_title("Overlay")
    axes[i,3].axis("off")

plt.tight_layout()
plt.savefig(RESULTS/"segmentation_examples.png", dpi=300)
plt.close()

avg_dice = np.mean(dice_scores)
avg_iou = np.mean(iou_scores)

with open(RESULTS/"dice_score.txt","w") as f:
    f.write(f"Average Dice: {avg_dice:.4f}")

with open(RESULTS/"iou_score.txt","w") as f:
    f.write(f"Average IoU: {avg_iou:.4f}")

print(f"Average Dice: {avg_dice:.4f}")
print(f"Average IoU : {avg_iou:.4f}")
print("Saved:")
print(" - segmentation_examples.png")
print(" - dice_score.txt")
print(" - iou_score.txt")