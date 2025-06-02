# 🧠 Medical AI Navigator: Awesome Medical Datasets, Loss Functions, Tools & More for Deep Learning & Machine Learning

A curated collection of publicly available medical datasets, loss functions, state-of-the-art methods, and tools across various modalities and anatomical regions, tailored for researchers and practitioners in deep learning (DL) and machine learning (ML). This repository aims to facilitate the development and benchmarking of algorithms in medical imaging and related fields.

---

📚 **Table of Contents**
*   [Whole Body](#whole-body)
*   [Brain](#brain)
*   [Head and Neck](#head-and-neck)
*   [Chest (Lungs & Thorax)](#chest-lungs--thorax)
*   [Abdomen](#abdomen)
*   [Heart](#heart)
*   [Bones](#bones)
*   [Endoscopy](#endoscopy)
*   [Retina](#retina)
*   [Skin](#skin)
*   [Microscopic Imaging (Histopathology)](#microscopic-imaging-histopathology)
*   [Imaging and Text](#imaging-and-text)
*   [Text](#text)
*   [Other Medical Awesome Collection Projects](#other-medical-awesome-collection-projects)
*   [Dataset Platforms](#dataset-platforms)
*   [Loss Functions](#loss-functions)
    *   [Segmentation Loss Functions for Medical Imaging](#segmentation-loss-functions-for-medical-imaging)
    *   [Classification Loss Functions for Medical Imaging](#classification-loss-functions-for-medical-imaging)
    *   [Object Detection Loss Functions for Medical Imaging](#object-detection-loss-functions-for-medical-imaging)
*   [Evaluation Benchmarks](#evaluation-benchmarks)
*   [SOTA Methods](#sota-methods)
    *   [Segmentation](#segmentation)
*   [Related Foundation Toolbox Projects](#related-foundation-toolbox-projects)
*   [Contributing](#contributing)
*   [License](#license)

---

## 🏃‍♂️ Whole Body

| Dataset Name        | Task                      | Modality   | # Subjects | Labels/Classes                                       | Description                                                                                                  | Link                                      |
| :------------------ | :------------------------ | :--------- | :--------- | :--------------------------------------------------- | :----------------------------------------------------------------------------------------------------------- | :---------------------------------------- |
| CT-ORG              | Organ Segmentation        | CT         | 140        | 6 (lungs, liver, kidneys, spleen, bladder, bones)    | 3D CT scans with annotations for six organ categories.                                                         | [GitHub](https://github.com/MIC-DKFZ/CT-ORG) |
| AutoPET             | Tumor Segmentation        | PET-CT     | 1,214      | 1 (tumor)                                            | Whole-body tumor segmentation in PET-CT images from FDG-PET/CT scans.                                        | [Grand Challenge](https://autopet.grand-challenge.org/) |
| TotalSegmentator    | Organ Segmentation        | CT         | 1,204      | 104 (various organs & anatomical structures)       | Comprehensive organ segmentation dataset with 104 classes.                                                     | [GitHub](https://github.com/wasserth/TotalSegmentator) |
| TotalSegmentator V2 | Organ Segmentation        | CT         | 1,823      | 117 (various organs & anatomical structures)       | Updated version with additional organ classes and more subjects.                                               | [Zenodo](https://zenodo.org/record/81totalsegmentatorv2) |
| TotalSegmentator MRI| Organ Segmentation        | MRI        | 298        | 56 (various organs & anatomical structures)        | MRI-based organ segmentation dataset.                                                                          | [Zenodo](https://zenodo.org/record/10005224) |
| ULS                 | Tumor Segmentation        | CT         | 38,842     | 1 (cancer)                                           | Large-scale whole-body tumor segmentation dataset from Universal Lesion Segmentation challenge.                | [Grand Challenge](https://uls.grand-challenge.org/) |
| AMOS 2022           | Abdominal Organ Segmentation | CT & MRI   | 600        | 15 (spleen, R/L kidney, gallbladder, esophagus, liver, stomach, aorta, IVC, pancreas, R/L adrenal gland, duodenum, bladder, prostate/uterus) | Abdominal Multi-Organ Segmentation challenge, CT and MRI data.                                                | [Grand Challenge](https://amos22.grand-challenge.org/) |

---

## 🧠 Brain

| Dataset Name          | Task                                     | Modality             | # Subjects | Labels/Classes                                               | Description                                                                                                                            | Link                                                           |
| :-------------------- | :--------------------------------------- | :------------------- | :--------- | :----------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------- |
| BraTS (2021-2023)     | Brain Tumor Segmentation                 | MRI (T1, T1c, T2, FLAIR) | ~2000+ (varies by year) | 3 (enhancing tumor, tumor core, whole tumor) / 4 (labels per region) | Multimodal Brain Tumor Segmentation Challenge. Data focuses on gliomas. Number of subjects increases with each challenge year.             | [CBICA](https://www.med.upenn.edu/cbica/brats-challenge/)        |
| ADNI                  | Alzheimer's Disease Classification/Progression | MRI, PET, Clinical | 1500+      | Normal, MCI, AD                                              | Alzheimer's Disease Neuroimaging Initiative. Longitudinal multimodal data including imaging, clinical, and genetic information.        | [ADNI](http://adni.loni.usc.edu/)                               |
| OASIS                 | Brain Aging, Alzheimer's Disease         | MRI                  | ~400 (OASIS-1), ~150 (OASIS-2), ~1000 (OASIS-3) | Healthy, AD, Cognitive status | Open Access Series of Imaging Studies. Longitudinal neuroimaging, clinical, and cognitive data for normal aging and Alzheimer's Disease. | [OASIS Brains](https://www.oasis-brains.org/)                  |
| IXI Dataset           | Brain Segmentation, Atlas Building       | MRI (T1, T2, PD, MRA, DWI) | ~600       | N/A (raw data for structure analysis)                        | Information eXtraction from Images. Brain MRI data from healthy adults.                                                                  | [IXI Dataset](https://brain-development.org/ixi-dataset/)     |
| ATLAS R2.0            | Stroke Lesion Segmentation               | MRI (T1, T2, FLAIR, DWI) | 655        | Stroke lesion                                                | Anatomical Tracings of Lesions After Stroke. Multi-site dataset with manual segmentations.                                             | [ATLAS](http://fcon_1000.projects.nitrc.org/indi/retro/atlas_r2.0.html) |
| HCP                   | Connectomics, Brain Parcellation         | MRI (dMRI, fMRI, T1, T2) | 1200       | N/A (extensive neuroimaging data)                          | Human Connectome Project. Aims to map human brain connectivity. Provides extensive imaging and behavioral data.                        | [Human Connectome](https://www.humanconnectome.org/study/hcp-young-adult) |

---

## 🤯 Head and Neck

| Dataset Name         | Task                                   | Modality | # Subjects | Labels/Classes                           | Description                                                                                      | Link                                                              |
| :------------------- | :------------------------------------- | :------- | :--------- | :--------------------------------------- | :----------------------------------------------------------------------------------------------- | :---------------------------------------------------------------- |
| MICCAI2024-AutoPETIII| Tumor Lesion Segmentation              | PET-CT   | 1,614      | Multiple (tumor lesions)                 | Multi-center, multi-tracer automatic tumor lesion segmentation in head & neck, and whole body PET-CT. | [Grand Challenge](https://autopet3.grand-challenge.org/)        |
| ImageCAS             | Coronary Artery Segmentation           | CTA      | 1,000      | Coronary arteries                        | Large-scale coronary artery segmentation dataset from Cardiac CTA. (Note: Heart-related but often involves neck vessels). | [GitHub](https://github.com/FunClipQA/ImageCAS)                  |
| Head-Neck-CT-Atlas   | Organ at Risk Segmentation             | CT       | 25         | 9 (brainstem, mandible, parotids, etc.)  | CT scans of head and neck cancer patients with manual OAR contours.                               | [TCIA](https://wiki.cancerimagingarchive.net/display/Public/Head-Neck-CT-Atlas) |
| PDDCA                | Oropharyngeal Cancer Outcome Prediction| CT, Clinical| 312      | Recurrence, Distant Metastasis         | Patient data with CT images and clinical info for predicting oropharyngeal cancer outcomes.        | [TCIA](https://wiki.cancerimagingarchive.net/display/Public/PDDCA) |

---

## 🫁 Chest (Lungs & Thorax)

| Dataset Name        | Task                                   | Modality | # Subjects  | Labels/Classes                                               | Description                                                                                                            | Link                                                        |
| :------------------ | :------------------------------------- | :------- | :---------- | :----------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------- |
| PadChest            | Multi-label Classification, Report Gen | X-ray    | ~67,000 (patients), ~160,000 (images) | 174 (radiological findings), 19 (differential diagnoses) | Large chest X-ray dataset with multi-label annotations and associated radiological reports.                                | [PhysioNet](https://physionet.org/content/padchest/1.0.0/) or [arXiv](https://arxiv.org/abs/1901.07441) |
| MedFM ChestDR2023   | Disease Classification                 | X-ray    | 4,848      | 19 (common chest abnormalities)                            | Classification of 19 common chest abnormalities from digital radiography.                                                | [GitHub](https://github.com/FudanMEDPMC/MedFM/tree/main/benchmark/ChestDR2023) |
| LIDC-IDRI           | Lung Nodule Detection & Segmentation   | CT       | 1,018      | Nodule outlines and characteristics (e.g., malignancy)     | Lung Image Database Consortium image collection. Thoracic CT scans with annotated lesions by multiple radiologists.        | [TCIA](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI) |
| LUNA16              | Lung Nodule Detection                  | CT       | 888        | Nodule locations (present/absent)                            | LUng Nodule Analysis 2016 challenge dataset, derived from LIDC-IDRI. Focuses on nodule detection.                      | [Grand Challenge](https://luna16.grand-challenge.org/Data/) |
| ChestX-ray8         | Disease Classification                 | X-ray    | 30,805 (patients), ~112,000 (images) | 8 (Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax) | Chest X-ray database from NIH, with 8 common thoracic disease labels mined from radiological reports.                   | [NIH CXR8](https://nihcc.app.box.com/v/ChestXray-NIHCC)   |
| MIMIC-CXR           | Multi-label Classification, Report Gen | X-ray    | ~227,000 (patients), ~377,000 (images) | 14 (common pathological conditions)                        | Large dataset of chest X-rays with free-text radiology reports and structured labels. Part of the MIMIC-IV project.       | [PhysioNet](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) |
| COVID-19 Radiography| COVID-19 Classification                | X-ray    | ~21,000    | COVID-19, Viral Pneumonia, Normal                          | A large, diverse dataset of chest X-ray images for COVID-19 detection.                                               | [Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) |
| RibFrac Challenge   | Rib Fracture Detection & Classification| CT       | 500 cases  | Fractured (yes/no), Displaced, Buckle, Nondisplaced      | Rib Fracture detection, segmentation, and classification challenge dataset.                                            | [Grand Challenge](https://ribfrac.grand-challenge.org/Dataset/) |

---

## 🍽 Abdomen

| Dataset Name     | Task                           | Modality | # Subjects | Labels/Classes                                            | Description                                                                                                             | Link                                                                  |
| :--------------- | :----------------------------- | :------- | :--------- | :-------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------- |
| AbdomenCT-1K     | Organ Segmentation             | CT       | 1,112      | Multiple (major abdominal organs)                         | Clinically diverse abdominal CT organ segmentation dataset with over 1000 patients.                                   | [Synapse](https://www.synapse.org/#!Synapse:syn27617043/wiki/617210) or [arXiv](https://arxiv.org/abs/2210.06808) |
| LiTS             | Liver & Liver Tumor Segmentation| CT       | 201        | Liver, Liver Tumor                                        | Liver Tumor Segmentation Challenge. Focuses on segmentation of liver and liver lesions.                                 | [LiTS Challenge](http://litsc.grand-challenge.org/data/)             |
| KiTS23           | Kidney & Kidney Tumor Segmentation | CT    | 489        | Kidney, Kidney Tumor, Cyst                                | Kidney and Kidney Tumor Segmentation Challenge for nephrectomy cases.                                                   | [KiTS23 GitHub](https://github.com/neheller/kits23)                     |
| CHAOS            | Abdominal Organ Segmentation   | CT & MRI | 40 (CT), 40 (MRI) | 4 (liver, R/L kidney, spleen) in MRI, multiple in CT | Combined (CT-MR) Healthy Abdominal Organ Segmentation Challenge. Provides T1-DUAL (In-Phase/Out-of-Phase) and T2-SPIR MR. | [Grand Challenge](https://chaos.grand-challenge.org/Data_Download/)  |
| MSD Pancreas     | Pancreas & Pancreatic Tumor Seg | CT      | 420        | Pancreas, Pancreatic Tumor                                | Part of the Medical Segmentation Decathlon focusing on pancreas and pancreatic tumor segmentation.                        | [Medical Decathlon](http://medicaldecathlon.com/)                     |

---

## ❤️ Heart

| Dataset Name      | Task                                               | Modality | # Subjects  | Labels/Classes                                      | Description                                                                                                            | Link                                                               |
| :---------------- | :------------------------------------------------- | :------- | :---------- | :-------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------- |
| EMIDEC            | Myocardial Infarction Assessment                   | MRI      | 100 (patients), 150 (scans) | Infarct, No-reflow, Normal Myocardium             | Evaluation of Myocardial Infarction from DElayed-enhancement Cardiac MRI. Contains expert segmentations.               | [GitHub](https://github.com/EMIDEC-Challenge/EMIDEC-Dataset)       |
| ACDC              | Cardiac Segmentation (LV, RV, Myo) & Disease Class. | MRI      | 100-150 (varies per release) | LV, RV, Myocardium at ED & ES; Disease state      | Automated Cardiac Diagnosis Challenge. Cine MRI for left/right ventricle and myocardium segmentation, disease classification. | [ACDC Challenge](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html) |
| M&Ms Challenge    | Multi-Vendor, Multi-Disease Cardiac Segmentation   | MRI      | 375         | LV, RV, Myocardium at ED & ES                     | Multi-Centre, Multi-Vendor & Multi-Disease Cardiac Image Segmentation Challenge. Tests domain generalization.           | [M&Ms Challenge](https://www.ub.edu/mnms/)                         |
| CAMUS             | Cardiac Ultrasound Segmentation & Quality Assessment | Echo     | 500         | LV Endocardium, LV Epicardium, LA Endocardium at ED & ES | Cardiac M-mode, 2-chamber, and 4-chamber echocardiography sequences with ground truth segmentations and image quality grades. | [CAMUS Dataset](https://www.creatis.insa-lyon.fr/Challenge/camus/) |

---

## 🦴 Bones

| Dataset Name   | Task                                      | Modality | # Subjects (or Images) | Labels/Classes                             | Description                                                                                                         | Link                                                                       |
| :------------- | :---------------------------------------- | :------- | :--------------------- | :----------------------------------------- | :------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------- |
| MURA           | Musculoskeletal Abnormality Detection     | X-ray    | ~14,863 (studies), ~40,561 (images) | Normal, Abnormal                           | Musculoskeletal Radiographs dataset of upper extremities (shoulder, humerus, elbow, forearm, wrist, hand).          | [Stanford ML Group](https://stanfordmlgroup.github.io/competitions/mura/)  |
| RibFrac        | Rib Fracture Detection & Segmentation     | CT       | 500 (cases)            | Rib fracture presence, type, displacement  | Dataset for rib fracture detection, segmentation, and classification from chest CT scans. See also in Chest section.   | [Grand Challenge](https://ribfrac.grand-challenge.org/Dataset/)           |
| VinDr-BoneMRI  | Knee MRI Abnormality Classification/Localization | MRI   | 1,150 (studies)         | 8 (ACL tear, Meniscal tear, etc.)           | Knee MRI dataset with abnormalities labeled by radiologists, including bounding boxes for lesions.                   | [PhysioNet](https://physionet.org/content/vindr-bonemri/1.0.0/)             |
| RSNA Bone Age  | Bone Age Assessment                       | X-ray    | ~12,600 (images)       | Bone age (months), Male/Female             | Pediatric hand X-rays for bone age assessment challenge.                                                              | [Kaggle (RSNA Bone Age)](https://www.kaggle.com/c/rsna-bone-age/data)       |
| VerSe          | Vertebral Segmentation & Labeling         | CT       | 374 (scans)            | Vertebrae labels (C1-L5, etc.) & centroids | Large-scale Vertebral Segmentation Challenge. Multi-site, multi-scanner CT data with vertebral body segmentations.  | [Grand Challenge](https://verse.grand-challenge.org/Data/)               |

---

## 🩺 Endoscopy

| Dataset Name     | Task                                                  | Modality        | # Subjects/Videos/Frames | Labels/Classes                                                | Description                                                                                                   | Link                                                                |
| :--------------- | :---------------------------------------------------- | :-------------- | :----------------------- | :------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------ |
| SurgVU24         | Surgical Instrument Identification, Phase Recognition | Endoscopy Video | - (Multiple procedures)  | Surgical instruments, Surgical phases                       | Identification of surgical instruments and phase recognition in surgeries. Video data.                         | [GitHub](https://github.com/SurgVU/SurgVU24Challenge)               |
| Kvasir-SEG       | Polyp Segmentation                                    | Endoscopy       | 1000 (images)            | Polyp                                                         | Gastrointestinal polyp image dataset with corresponding segmentation masks.                                    | [Kvasir Dataset Page](https://datasets.simula.no/kvasir-seg/)       |
| Kvasir-Capsule   | Lesion Detection, Anatomical Landmark Recognition     | Capsule Endoscopy | ~44,000 (images)         | Various findings (polyps, bleeding, inflammation), Landmarks | Images from capsule endoscopy, annotated for various findings and anatomical landmarks.                        | [Kvasir Dataset Page](https://datasets.simula.no/kvasir-capsule/)   |
| HyperKvasir      | Image Classification, Segmentation                    | Endoscopy       | ~110,000 (images & frames from 374 videos) | ~23 classes (pathological findings, anatomical landmarks)   | Large gastrointestinal endoscopy dataset with labeled images and segmented frames for various clinical findings. | [HyperKvasir Page](https://hyperkvasir.simula.no/)                  |
| EAD2020          | Endoscopic Artefact Detection                         | Endoscopy Video | 22 (sequences)           | 7 artefact classes (blur, saturation, contrast etc.)        | Challenge dataset for detecting various visual artifacts in colonoscopy videos.                                | [Grand Challenge](https://ead2020.grand-challenge.org/EAD2020/)     |

---

## 👁 Retina

| Dataset Name    | Task                                        | Modality    | # Subjects/Images | Labels/Classes                             | Description                                                                                                | Link                                                                   |
| :-------------- | :------------------------------------------ | :---------- | :---------------- | :----------------------------------------- | :--------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------- |
| LES-AV          | Artery/Vein Segmentation                    | Fundus      | 22 (images)       | Artery, Vein                               | Fundus images with artery and vein annotations from the DRIVE dataset.                                     | [GitHub](https://github.com/ShengWelson/LES-AV-Retinal-Vessel-Segmentation) |
| DRIVE           | Vessel Segmentation                         | Fundus      | 40 (images)       | Vessel/Non-vessel                          | Digital Retinal Images for Vessel Extraction. Widely used benchmark for retinal vessel segmentation.        | [DRIVE Page](https://drive.grand-challenge.org/Data/)                 |
| STARE           | Vessel Segmentation, Pathology Detection    | Fundus      | 20 (images)       | Vessel/Non-vessel, Pathologies             | STructured Analysis of the REtina. Fundus images with vessel segmentations and clinical labels.           | [STARE Project](http://www.ces.clemson.edu/~ahoover/stare/)           |
| Messidor        | Diabetic Retinopathy Grading                | Fundus      | 1200 (images)     | Retinopathy grade (0-3), Risk of macular edema | Fundus images for diabetic retinopathy grading.                                                              | [Messidor Page](https://www.adcis.net/en/third-party/messidor/)     |
| IDRiD           | DR Grading, Lesion Segmentation             | Fundus      | 516 (images)      | DR grades, Lesions (MA, HE, EX, SE)        | Indian Diabetic Retinopathy Image Dataset. Offers DR grading and segmentation of microaneurysms, hemorrhages, etc. | [IDRiD Challenge](https://idrid.grand-challenge.org/Data/)            |
| OCTA-500        | OCT Angiography Vessel Segmentation         | OCT-A       | 500 (volumes)     | Superficial, Deep, Choriocapillaris Layers | Optical Coherence Tomography Angiography for retinal vessel segmentation in different layers.               | [OCTA-500](http://www.medimaging.net/octa-500/)                   |

---

## 🧴 Skin

| Dataset Name | Task                                      | Modality              | # Subjects/Images | Labels/Classes                                         | Description                                                                                                                  | Link                                                                            |
| :----------- | :---------------------------------------- | :-------------------- | :---------------- | :----------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------ |
| ISIC Archive | Skin Lesion Analysis (Segment., Classif.) | Dermoscopy, Clinical  | 100,000+ (images, varies by challenge year) | Melanoma, Nevus, BCC, AKIEC, BKL, DF, VASC, SCC | International Skin Imaging Collaboration. Large archive of skin lesion images for melanoma detection and other tasks. Hosts annual challenges. | [ISIC Archive](https://www.isic-archive.com/#!/topWithHeader/tightContentTop) |
| HAM10000     | Skin Lesion Classification                | Dermoscopy, Histology | ~7,500 (patients), ~10,000 (images) | 7 (AKIEC, BCC, BKL, DF, MEL, NV, VASC)             | "Human Against Machine with 10000 training images." Multi-class skin lesion classification dataset.                          | [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) |
| Derm7pt      | Skin Lesion Classification                | Dermoscopy, Clinical  | ~2,000 (images)   | Melanoma, Nevus, Seborrheic Keratosis; 7-point checklist features | Dataset containing dermoscopic images and clinical images with 7-point checklist information.                               | [Derm7pt Info](https://derm.cs.sfu.ca/Welcome.html)                             |
| PAD-UFES-20  | Skin Disease Classification               | Clinical Images       | ~2,298 (images)   | 6 (Acne, Eczema, Psoriasis, etc.), 20 anatomical sites | Mobile phone acquired images of skin diseases, covering 20 different anatomical sites and 6 disease classes.                  | [IEEE DataPort](https://ieee-dataport.org/open-access/pad-ufes-20-skin-lesion-classification) |

---

## 🔬 Microscopic Imaging (Histopathology)

| Dataset Name      | Task                                         | Modality            | # Subjects/Images/Patches | Labels/Classes                                  | Description                                                                                                 | Link                                                                      |
| :---------------- | :------------------------------------------- | :------------------ | :------------------------ | :---------------------------------------------- | :---------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------ |
| CoNIC2022         | Nuclei Instance Segmentation & Classification| Histology (H&E)     | 4,981 (patches)           | 6 nuclei types (Neutrophil, Epithelial, etc.) | Colon Nuclei Identification and Counting Challenge. Focuses on segmentation and classification of nuclei in H&E stained colorectal adenocarcinoma images. | [CoNIC GitHub](https://conic-challenge.github.io/2022/)                 |
| PathMMU           | Multimodal Understanding                     | Pathology (WSI, Text) | - (Benchmark)             | Multiple (Pathology QA, Report Gen)           | Benchmark dataset for evaluating multimodal models in pathology, often linking WSI to text-based questions or reports. | [GitHub](https://github.com/PathMMU/PathMMU)                              |
| Camelyon16 / 17   | Lymph Node Metastasis Detection              | Histology (WSI)     | ~400 (Cam16), ~1000 (Cam17) WSI | Metastasis, Normal                            | Grand Challenge for breast cancer metastasis detection in whole-slide images of lymph nodes. Camelyon17 added patient-level survival prediction. | [Camelyon16](https://camelyon16.grand-challenge.org/Data/) / [Camelyon17](https://camelyon17.grand-challenge.org/Data/) |
| TUPAC16           | Tumor Proliferation Assessment               | Histology (WSI)     | 500 (training), 321 (test) images | Mitotic figures, Tumor cellularity scores       | TUmor Proliferation Assessment Challenge. Grading breast cancer histopathology based on mitotic count and cellularity. | [TUPAC16](http://tupac.tue-image.nl/node/3)                               |
| GlaS@MICCAI2015   | Gland Segmentation in Colon Histology        | Histology (H&E)     | 165 (images)              | Gland                                           | Gland Segmentation in Colon Histology Images Challenge. Focuses on benign and malignant glands.              | [GlaS Challenge](https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/download/) |
| BCSS              | Breast Cancer Semantic Segmentation          | Histology (H&E)     | 151 (ROIs from WSIs)      | 6 (Tumor, Stroma, Inflammatory, etc.)           | Breast Cancer Semantic Segmentation dataset with pixel-level annotations for various tissue types.          | [BCSS TCIA](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52756444) |

---

## 🖼️📝 Imaging and Text

| Dataset Name      | Task                                            | Modality                      | # Subjects/Studies | Labels/Classes                              | Description                                                                                                            | Link                                                               |
| :---------------- | :---------------------------------------------- | :---------------------------- | :----------------- | :------------------------------------------ | :--------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------- |
| MedFMC            | Disease Classification, Foundation Model Eval   | Multiple (X-Ray, CT, MRI, etc.) | 22,349             | 5 (disease states for benchmark task)       | A large-scale Multi-center, Multi-modal, Multi-task dataset for foundation model evaluation in medicine.             | [GitHub](https://github.com/FudanMEDPMC/MedFMC)                      |
| ROCO              | Radiology Objects in COntext                    | Radiology Images, Captions    | ~81,000 (images)   | N/A (Image-caption pairs)                   | Dataset of radiology images with corresponding captions. Useful for image captioning, VQA.                             | [ROCO Dataset](https://github.com/flaviagiamber/ROCO)               |
| MedVQA            | Medical Visual Question Answering               | Radiology Images, QA Pairs    | ~15,000 (VQA pairs) | N/A (Question-Answer pairs based on images) | Collection of medical images (X-rays, CTs) with clinically relevant questions and answers.                             | [MedVQA (SLAKE)](https://www.med-vqa.com/slake/)                  |
| MIMIC-CXR-Report  | Report Generation / Summarization               | X-ray, Text Reports           | ~227,000 (studies) | N/A (Image-Report pairs)                    | Chest X-rays linked to free-text radiology reports. Part of MIMIC-CXR. (Also see Chest section).                     | [PhysioNet](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)      |
| RadQA             | Question Answering on Radiology Reports         | Text Reports                  | ~300,000           | Answers to clinical questions               | Question answering dataset based on a large corpus of radiology reports, focusing on clinical information extraction. | [Stanford AIMI](https://stanfordaimi.azurewebsites.net/datasets/rubrique_10) |
| OpenI             | Image-Text Retrieval, Report Generation       | Chest X-ray, Text Reports     | ~3,900 (patients), ~7,400 (images) | N/A (Image-Report pairs)                  | Open-access database of de-identified chest X-rays with associated reports from Indiana University Health.           | [OpenI](https://openi.nlm.nih.gov/gridquery)                         |

---

## 📄 Text

| Dataset Name    | Task                                       | Modality | # Records/Questions | Labels/Classes                       | Description                                                                                             | Link                                                                   |
| :-------------- | :----------------------------------------- | :------- | :------------------ | :----------------------------------- | :------------------------------------------------------------------------------------------------------ | :--------------------------------------------------------------------- |
| MedQA           | Medical Question Answering                 | Text     | ~12,700 (USMLE style Qs) | Multiple-choice options (Correct answer) | Medical question and answer dataset in multiple-choice format, often from USMLE-style questions.          | [GitHub (MedQA)](https://github.com/biusante/medqa) or [Original GitHub](https://github.com/jind11/MedQA) |
| PubMedQA        | Biomedical Question Answering              | Text     | ~211k (pairs)       | Yes/No/Maybe                         | Question answering dataset where answers are derived from abstracts in PubMed.                           | [PubMedQA](https://pubmedqa.github.io/)                                 |
| MedNLI          | Natural Language Inference                 | Text     | ~14,000 (pairs)     | Entailment, Contradiction, Neutral   | A dataset for Natural Language Inference based on clinical notes from MIMIC-III.                          | [PhysioNet](https://physionet.org/content/mednli/1.0.0/)                 |
| emrQA           | Question Answering on EHR Notes            | Text     | ~400k (QA pairs)    | Answers from EHR text                | Question answering dataset focused on querying de-identified electronic health record notes.             | [emrQA GitHub](https://github.com/panushri/emrQA)                      |
| NCBI Disease    | Disease Name Recognition & Normalization   | Text     | 793 (abstracts)     | Disease mentions, Concept IDs        | Corpus of PubMed abstracts annotated for disease names and normalized to MeSH vocabulary.                   | [NCBI Disease Corpus](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/) |
| ClinicalTrials.gov| Various NLP tasks (summarization, IE)      | Text     | Millions (studies)  | Various fields per study             | Database of clinical studies. Can be used for information extraction, summarization, cohort selection.    | [ClinicalTrials.gov](https://clinicaltrials.gov/api/)                 |

---

## 🔗 Other Medical Awesome Collection Projects

*   [SegLossOdyssey](https://github.com/SegLossOdyssey/SegLossOdyssey): A collection of loss functions for medical image segmentation.
*   [SOTA-MedSeg](https://github.com/SOTA-MedSeg/SOTA-MedSeg): SOTA medical image segmentation methods based on various challenges.
*   [awesome-transformers-in-medical-imaging](https://github.com/freyja719/awesome-transformers-in-medical-imaging): A collection of resources on applications of Transformers in Medical Imaging.
*   [awesome-gan-for-medical-imaging](https://github.com/xiehongen/awesome-gan-for-medical-imaging): Awesome GAN for Medical Imaging.
*   [Awesome-Diffusion-Models-in-Medical-Imaging](https://github.com/amirhossein-kz/Awesome-Diffusion-Models-in-Medical-Imaging): Diffusion Models in Medical Imaging (Published in Medical Image Analysis Journal).
*   [Awesome-CLIP-in-Medical-Imaging](https://github.com/UsmanAnwar-byte/Awesome-CLIP-in-Medical-Imaging): A Survey on CLIP in Medical Imaging.
*   [awesome-multimodal-in-medical-imaging](https://github.com/InterestMat/awesome-multimodal-in-medical-imaging): A collection of resources on applications of multi-modal learning in medical imaging.
*   [awesome-medical-vision-language-models](https://github.com/CV_NLP_ALLSTAR/awesome-medical-vision-language-models): A collection of resources on Medical Vision-Language Models.
*   [Awesome-Medical-Large-Language-Models](https://github.com/ferris18/Awesome-Medical-Large-Language-Models): Curated papers on Large Language Models in Healthcare and Medical domain.
*   [awesome-healthcare](https://github.com/kakoni/awesome-healthcare): Curated list of awesome open source healthcare software, libraries, tools and resources.
*   [awesome_Chinese_medical_NLP](https://github.com/GanjinZero/awesome_Chinese_medical_NLP): 中文医学NLP公开资源整理：术语集/语料库/词向量/预训练模型/知识图谱/命名实体识别/QA/信息抽取/模型/论文/etc. (Chinese Medical NLP Open Resources)
*   [Data-Centric-FM-Healthcare](https://github.com/NamrataBonde/Data-Centric-FM-Healthcare): A survey on data-centric foundation models in computational healthcare.

---

## 🗂️ Dataset Platforms

You can search for more medical datasets not included in this project on these websites:

*   [Grand Challenges](https://grand-challenge.org/): A platform for end-to-end development of machine learning solutions in biomedical imaging.
*   [Kaggle](https://www.kaggle.com/datasets): One of the largest AI & ML community.
*   [TCIA (The Cancer Imaging Archive)](https://www.cancerimagingarchive.net/): A service which de-identifies and hosts a large archive of medical images of cancer accessible for public download.
*   [Synapse](https://www.synapse.org/): A platform for supporting scientific collaborations centered around shared biomedical data sets.
*   [Medical Segmentation Decathlon](http://medicaldecathlon.com/): The MSD challenge tests the generalisability of machine learning algorithms when applied to 10 different semantic segmentation tasks.
*   [CodaLab Competitions](https://codalab.lisn.upsaclay.fr/): An open-source web-based platform that enables researchers, developers, and data scientists to collaborate, with the goal of advancing research fields where machine learning and advanced computation is used.
*   [Tianchi (天池)](https://tianchi.aliyun.com/dataset): Tianchi is a developer competition platform under Alibaba Cloud. (Chinese)
*   [OpenDataLab](https://opendatalab.com/): China Big Model Corpus Data Alliance open source data service designated platform, provides high-quality open data sets for large models. (Chinese)

---

## 📉 Loss Functions

This section outlines commonly used loss functions for various medical image analysis tasks. Choosing the right loss function is crucial for model performance.

### 💠 Segmentation Loss Functions for Medical Imaging

| Loss Function Name     | Description                                                                        | Common Use Cases                                                        | Notes                                                                               |
| :--------------------- | :--------------------------------------------------------------------------------- | :---------------------------------------------------------------------- | :---------------------------------------------------------------------------------- |
| Dice Loss              | Measures overlap between prediction and ground truth. Robust to class imbalance.   | Organ segmentation, lesion segmentation                                 | Often preferred over cross-entropy for imbalanced segmentation tasks.               |
| Jaccard/IoU Loss       | Similar to Dice Loss, based on the Intersection over Union metric.                 | Organ segmentation, lesion segmentation                                 | Similar performance characteristics to Dice Loss.                                   |
| Tversky Loss           | Generalization of Dice Loss that allows weighting false positives and negatives.   | Imbalanced datasets, situations requiring penalizing FP/FN differently. | Requires careful tuning of α and β parameters.                                      |
| Focal Loss             | Modification of cross-entropy to handle class imbalance by down-weighting easily classified samples. | Object detection and segmentation with highly imbalanced classes.       | Helps focus on hard-to-classify samples.                                            |
| Cross-Entropy Loss (Pixel-wise) | Classifies each pixel independently, common for multi-class pixel-wise segmentation. | Semantic segmentation                                                   | Can be sensitive to class imbalance. Works well for balanced segmentation.        |
| Combo Loss             | Combination of Dice Loss and a modified Cross-Entropy.                             | Imbalanced class segmentation tasks.                                    | Aims to combine the benefits of both loss functions.                              |
| Lovász-Softmax Loss    | Direct optimization of the Jaccard index for multi-class segmentation problems.  | Multi-class segmentation, especially for small objects or complex shapes. | Can be computationally expensive on large datasets.                               |
| Boundary Loss          | Focuses on accurately segmenting boundaries by penalizing discrepancies at region contours. | Tasks where precise boundary delineation is critical.                   | Can be combined with region-based losses like Dice or Cross-Entropy.              |
| Hausdorff Distance Loss| Measures the maximum distance between the predicted segmentation and the ground truth. | Segmentation tasks requiring high shape similarity.                       | Sensitive to outliers, differentiable versions exist.                             |

### 🎯 Classification Loss Functions for Medical Imaging

| Loss Function Name             | Description                                                                                           | Common Use Cases                                                        | Notes                                                                         |
| :----------------------------- | :---------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------- | :---------------------------------------------------------------------------- |
| Binary Cross-Entropy (BCE)   | For binary classification tasks (e.g., disease presence/absence). Measures discrepancy between model prediction and true label. | Disease presence/absence detection, lesion malignancy classification  | Output should be passed through a sigmoid function.                         |
| Categorical Cross-Entropy    | For multi-class, single-label classification tasks (e.g., classifying an image into one of several disease types). | Disease type classification, tissue type classification               | Output should be passed through a softmax function.                         |
| Focal Loss                     | An improvement over BCE or Categorical Cross-Entropy for imbalanced datasets. Focuses on hard samples.    | Classification with imbalanced disease prevalence                       | The focusing parameter γ can be tuned.                                      |
| Hinge Loss                     | Primarily used with Support Vector Machines (SVMs), aims to find the maximum margin classifier.         | Specific machine learning approaches                                    | Less common for neural networks but can be useful in certain contexts.        |
| Sparse Categorical Cross-Entropy | Same as Categorical Cross-Entropy but when true labels are provided as integers (not one-hot encoded). | Multi-class classification with integer labels                          | Can reduce memory usage.                                                    |
| Weighted Cross-Entropy         | Assigns different weights to classes in the cross-entropy calculation, useful for class imbalance.        | Disease classification where some classes are rare.                       | Weights need to be chosen carefully (e.g., inverse class frequency).        |

### ⭕ Object Detection Loss Functions for Medical Imaging

| Loss Function Name     | Description                                                                                           | Common Use Cases                                                    | Notes                                                                           |
| :-------------------- | :---------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------ | :------------------------------------------------------------------------------ |
| Smooth L1 Loss        | Used for bounding box regression. Less sensitive to outliers than L2 Loss.                            | Lesion localization, medical device detection                       | Controls the transition between L1 and L2 loss.                               |
| Intersection over Union (IoU) Loss / Generalized IoU (GIoU) Loss | Measures the overlap between the predicted bounding box and the ground truth. GIoU improves upon cases with no overlap. | Improving localization accuracy of detected objects                 | Common evaluation metric also used as a loss. GIoU, DIoU, CIoU are variants. |
| Focal Loss            | Used with one-stage detectors to down-weight the loss assigned to well-classified negative examples.  | Tumor or nodule detection using one-stage detectors like RetinaNet  | Helps manage foreground-background class imbalance.                           |
| Cross-Entropy Loss (for classification part) | To perform class classification for each proposed bounding box within object detection frameworks.          | Assigning a class label to each detected object                     | Often combined with a bounding box regression loss.                           |
| Distance-IoU (DIoU) Loss / Complete-IoU (CIoU) Loss | Improves upon IoU loss by considering the central point distance and aspect ratio of bounding boxes. CIoU also considers aspect ratio consistency. | More accurate and faster bounding box regression                   | Designed for better convergence speed and performance.                        |

---

## 📊 Evaluation Benchmarks

*Note: Many of the datasets collected in this project are benchmarks on the data platform. This section collects comprehensive benchmarks specifically used to evaluate model performance.*

| Benchmark     | Organization     | Paper                                      | Website                                 |
| :------------ | :--------------- | :----------------------------------------- | :-------------------------------------- |
| MedBench      | Shanghai AI Lab  | [arXiv](<Link to MedBench arXiv Paper>)      | [Project Link](<Link to MedBench Project>) |
| OmniMedVQA    | Shanghai AI Lab  | [arXiv](<Link to OmniMedVQA arXiv Paper>)  | [GitHub stars](<Link to OmniMedVQA GitHub>) |
| A-Eval        | Shanghai AI Lab  | [arXiv](<Link to A-Eval arXiv Paper>)      | [GitHub stars](<Link to A-Eval GitHub>)    |
| GMAI-MMBench  | Shanghai AI Lab  | [arXiv](<Link to GMAI-MMBench arXiv Paper>)| [GitHub stars](<Link to GMAI-MMBench GitHub>) |
| Touchstone    | JHU              | [arXiv](<Link to Touchstone arXiv Paper>)  | [GitHub stars](<Link to Touchstone GitHub>) |

*(Please replace `<Link to ...>` with the actual URLs.)*

---

## 🏆 SOTA Methods

### Segmentation

| Model             | Organization     | Paper                                    | Github                                  |
| :---------------- | :--------------- | :--------------------------------------- | :-------------------------------------- |
| nnU-Net           | DKFZ             | [arXiv](<Link to nnU-Net arXiv Paper>)   | [GitHub stars](<Link to nnU-Net GitHub>) |
| MedNeXt           | DKFZ             | [arXiv](<Link to MedNeXt arXiv Paper>)   | [GitHub stars](<Link to MedNeXt GitHub>) |
| STU-Net           | Shanghai AI Lab  | [arXiv](<Link to STU-Net arXiv Paper>)   | [GitHub stars](<Link to STU-Net GitHub>) |
| U-Net & CLIP      | CityU            | [arXiv](<Link to U-Net & CLIP arXiv Paper>) | [GitHub stars](<Link to U-Net & CLIP GitHub>) |
| UNETR             | NVIDIA           | [arXiv](<Link to UNETR arXiv Paper>)     | [GitHub stars](<Link to UNETR GitHub>)   |
| Swin UNETR        | NVIDIA           | [arXiv](<Link to Swin UNETR arXiv Paper>)| [GitHub stars](<Link to Swin UNETR GitHub>)|
| Swin UNETR & CLIP | CityU            | [arXiv](<Link to Swin UNETR & CLIP arXiv Paper>) | [GitHub stars](<Link to Swin UNETR & CLIP GitHub>) |
| MedFormer         | Rutgers          | [arXiv](<Link to MedFormer arXiv Paper>) | [GitHub stars](<Link to MedFormer GitHub>)|
| UniSeg            | NPU              | [arXiv](<Link to UniSeg arXiv Paper>)    | [GitHub stars](<Link to UniSeg GitHub>)  |
| Diff-UNet         | HKUST            | [arXiv](<Link to Diff-UNet arXiv Paper>) | [GitHub stars](<Link to Diff-UNet GitHub>)|
| LHU-Net           | UR               | [arXiv](<Link to LHU-Net arXiv Paper>)   | [GitHub stars](<Link to LHU-Net GitHub>) |
| NexToU            | HIT              | [arXiv](<Link to NexToU arXiv Paper>)    | [GitHub stars](<Link to NexToU GitHub>)  |
| SegVol            | BAAI             | [arXiv](<Link to SegVol arXiv Paper>)    | [GitHub stars](<Link to SegVol GitHub>)  |
| UNesT             | NVIDIA           | [arXiv](<Link to UNesT arXiv Paper>)     | [GitHub stars](<Link to UNesT GitHub>)   |
| SAM-Adapter       | Duke             | [arXiv](<Link to SAM-Adapter arXiv Paper>)| [GitHub stars](<Link to SAM-Adapter GitHub>)|

*(Please replace `<Link to ...>` with the actual URLs. "GitHub stars" might refer to the direct GitHub link or a badge showing stars; adjust as needed.)*

---

## 🛠️ Related Foundation Toolbox Projects

Many of the papers that use the datasets collected in this project have their code written based on the following Github code projects.

| Projects        | Organization    | Paper                                   | Github                                |
| :-------------- | :-------------- | :-------------------------------------- | :------------------------------------ |
| nnU-Net         | DKFZ            | [arXiv](<Link to nnU-Net arXiv Paper>)  | [GitHub stars](<Link to nnU-Net GitHub>) |
| MONAI           | NVIDIA          | [arXiv](<Link to MONAI Paper/Website>)  | [GitHub stars](<Link to MONAI GitHub>)  |
| SAM             | META            | [arXiv](<Link to SAM Paper>)            | [GitHub stars](<Link to SAM GitHub>)    |
| MMSegmentation  | Shanghai AI Lab | -                                       | [GitHub stars](<Link to MMSegmentation GitHub>) |
| MMPretrain      | Shanghai AI Lab | -                                       | [GitHub stars](<Link to MMPretrain GitHub>) |

*(Please replace `<Link to ...>` with the actual URLs. For projects like MONAI or MMSegmentation that might not have a single defining "paper" in the same way as a model, you might link to their main documentation, GitHub, or a general overview paper if available.)*

---

🤝 **Contributing**

Contributions are welcome! Please feel free to submit a pull request or open an issue to add new datasets, loss functions, tools, correct information, or improve existing entries. Ensure that any dataset added is publicly available and provide all relevant details as per the existing table structure.

When contributing, please try to provide:
*   **Resource Name:** Official or widely recognized name (e.g., Dataset Name, Loss Function Name, Tool Name).
*   **Task (for datasets/loss functions):** The primary ML/DL task(s) the resource is used for (e.g., Segmentation, Classification, Detection, Regression, Generation, VQA).
*   **Modality (for datasets):** The type of medical data (e.g., CT, MRI, X-ray, Histology, Text, Endoscopy).
*   **# Subjects (or Images/Scans/Patches - for datasets):** Approximate number of unique subjects or data points. Specify if it's images, scans, etc.
*   **Labels/Classes (for datasets/loss functions):** Key labels or classes annotated/predicted (e.g., specific organs, disease types, findings).
*   **Description:** A brief overview of the resource, its origin, and its purpose.
*   **Link:** A direct link to the resource download page, official website, primary publication, or GitHub repository.

---

📄 **License**

This repository and its contents are licensed under the [MIT License](LICENSE.md). Please ensure you comply with the individual licenses of each dataset and tool listed, as they may vary.

---

*Note: The number of subjects, images, and labels/classes for datasets are based on available information at the time of listing and may vary depending on dataset versions or specific subsets used. For detailed descriptions, terms of use, and access, please refer to the provided links.*
