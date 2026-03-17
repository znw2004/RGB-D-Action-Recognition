# RGB-D Action Recognition Based on Cross-Modal Fusion Transformer (THU-READ Dataset)

This repository provides the implementation of the paper *Cross-Modal Transformer for RGB-D Action Recognition*.

The method leverages **RGB and Depth modalities** with a **Transformer architecture** for cross-modal spatiotemporal modeling, achieving robust action recognition in complex indoor environments.

---

## 🔥 Key Features

Compared with the original DSCMT method, the main improvements are:

1. Changed the original dynamic image-based input to **frame-based input**, preserving fine-grained spatiotemporal information.  
2. The model is trained and evaluated **only on the THU-READ dataset**.  
3. Validated the effectiveness of frame-level input under a unified evaluation protocol in complex indoor scenarios.  
4. Supports cross-modal fusion modeling of RGB + Depth data.

---

## 🧱 Project Structure
DSCMT-main/
├── main.py # Training entry point
├── test_models.py # Testing entry point
├── DSCMT.py # Main model
├── dataset.py # Data loader
├── transforms.py # Data preprocessing
├── opts.py # Configuration options
├── resnet.py / vgg.py # Backbone networks
│
├── ops/ # Core modules
├── utils/ # Utility functions
├── tools/ # Dataset preprocessing scripts
├── train_test_files/ # Train/test split files
├── scores/ # Testing results


---


## ⚙️ Environment Requirements

* Python 3.7+  
* PyTorch 1.10+  
* opencv-python  
* Pillow  

Install dependencies:

```bash
pip install -r requirements.txt


----


📂 Dataset

This project uses only the THU-READ dataset:

The dataset is not included in this repository.

Please download and organize it manually.

📄 Data Preparation

Generate the training/testing list files:
python tools/dataset_protocol_train_test_list.py
After execution, you will obtain:
train_universal.txt
val_universal.txt
test_universal.txt


🚀 Training
1. Set modality environment variable (required)
$env:APPEAR_PAIR="RGBD"
2. Start training
python main.py thu-read Appearance path_of_train_list.txt \
  --arch resnet50 \
  --num_segment 1 \
  --lr 0.001 \
  --lr_steps 25 50 \
  --epochs 60 \
  -b 64 \
  --snapshot_pref saved_model_name \
  --val_list path_of_val_list.txt \
  --gpus 0
📌 Notes

Appearance: indicates frame-based input (RGB + Depth)

--num_segment 1: core modification for frame-level input

--gpus 0: single-GPU training

🧪 Testing
python test_models.py thu Appearance path_of_test_list.txt \
  path_of_model.pth.tar \
  --arch resnet50 \
  --test_segments 1 \
  --save_scores path_of_save_scores \
  --gpus 0 \
  --workers 0
📌 Example (Local Path)
python test_models.py thu Appearance E:\transformer_code\DSCMT-main\train_test_files\val_universal.txt \
E:\transformer_code\DSCMT-main\test_app_appearance_model_best.pth.tar \
--arch resnet50 \
--test_segments 1 \
--save_scores E:\transformer_code\DSCMT-main\scores\thu_rgbd \
--gpus 0 \
--workers 0
📊 Output

Testing results will be saved in:

scores/

Including:

Classification results

Confidence scores

Visualization results (partial)

📌 Important Notes

This project uses frame-based input.

Different from the original DSCMT's dynamic image input.

All experiments are conducted on the THU-READ dataset.

Default modality: RGB + Depth (RGBD).
