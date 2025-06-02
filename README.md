# 🩺 Awesome Medical Datasets for Deep Learning and Machine Learning

A comprehensive collection of publicly available medical datasets, organized by organ system, to facilitate research and development in medical imaging and related fields.

---

## 🧠 Brain Datasets

| Dataset Name | Task(s) | Modality | Subjects | Labels | Description | Access Link |
|--------------|---------|----------|----------|--------|-------------|-------------|
| **BraTS** | Tumor Segmentation | MRI (T1, T1c, T2, FLAIR) | ~750 | Tumor sub-regions (enhancing tumor, tumor core, whole tumor) | Benchmark dataset for brain tumor segmentation. | [Link](https://www.med.upenn.edu/sbia/brats2018/data.html) |
| **ADNI** | Disease Progression, Classification | MRI, PET, CSF biomarkers | >1,000 | Alzheimer's stages (CN, MCI, AD) | Longitudinal study for Alzheimer's disease research. | [Link](http://adni.loni.usc.edu/) |
| **Human Connectome Project (HCP)** | Brain Connectivity Mapping | MRI (Structural, Functional) | 1,200 | Brain connectivity metrics | High-resolution imaging for mapping human brain connectivity. | [Link](https://www.humanconnectome.org/) |
| **MRBrainS** | Brain Tissue Segmentation | MRI (T1, T1 IR, FLAIR) | 20 | Gray Matter, White Matter, CSF | Segmentation of brain tissues in elderly subjects. | [Link](https://mrbrains18.isi.uu.nl/) |
| **iSeg-2017** | Infant Brain Segmentation | MRI (T1, T2) | 10 | Brain tissues | Segmentation challenge for 6-month-old infant brains. | [Link](http://iseg2017.web.unc.edu/) |
| **SynthStrip** | Skull-Stripping | MRI, CT, PET | >600 | Brain masks | Dataset for evaluating brain extraction algorithms. | [Link](https://github.com/beamandrew/medical-data) |
| **MOOD** | Out-of-Distribution Detection | MRI, CT | 800 (Brain), 550 (Abdomen) | Anomaly detection labels | Dataset for evaluating OOD detection methods. | [Link](https://neuro-ml.github.io/amid/latest/datasets-api/) |

---

## ❤️ Heart Datasets

| Dataset Name | Task(s) | Modality | Subjects | Labels | Description | Access Link |
|--------------|---------|----------|----------|--------|-------------|-------------|
| **MSD Cardiac (Task02)** | Left Atrium Segmentation | MRI | 30 | Left atrium masks | Part of the Medical Segmentation Decathlon. | [Link](https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/MSD_Cardiac.md) |
| **Automated Cardiac Diagnosis Challenge (ACDC)** | Cardiac Structure Segmentation | MRI | 150 | LV, RV, Myocardium | Segmentation and diagnosis of cardiac structures. | [Link](https://www.creatis.insa-lyon.fr/Challenge/acdc/) |
| **UK Biobank Cardiac** | Cardiac Imaging Analysis | MRI | 5,000+ | Cardiac measurements | Large-scale dataset for cardiac imaging research. | [Link](https://www.ukbiobank.ac.uk/) |
| **Sunnybrook Cardiac Data** | LV Segmentation | MRI | 45 | LV contours | Dataset for evaluating LV segmentation algorithms. | [Link](https://www.cardiacatlas.org/studies/sunnybrook-cardiac-data/) |

---

## 🫁 Lung Datasets

| Dataset Name | Task(s) | Modality | Subjects | Labels | Description | Access Link |
|--------------|---------|----------|----------|--------|-------------|-------------|
| **ChestX-ray8** | Disease Classification, Localization | X-ray | 32,717 | 8 thoracic disease labels | Large-scale chest X-ray dataset with disease labels. | [Link](https://nihcc.app.box.com/v/ChestXray-NIHCC) |
| **CheXpert** | Disease Classification | X-ray | 65,240 | 14 observations with uncertainty labels | Dataset with uncertainty labels for chest radiographs. | [Link](https://stanfordmlgroup.github.io/competitions/chexpert/) |
| **MIMIC-CXR-JPG** | Disease Classification | X-ray | 227,827 studies | 14 labels derived from reports | Large chest radiograph dataset with associated reports. | [Link](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) |
| **PadChest** | Multi-label Classification | X-ray | 67,000 | 174 findings, 19 diagnoses | High-resolution chest X-ray dataset with detailed labels. | [Link](http://bimcv.cipf.es/bimcv-projects/padchest/) |
| **LIDC-IDRI** | Lung Nodule Detection | CT | 1,018 | Nodule annotations | Dataset for lung nodule detection and classification. | [Link](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI) |

---

## 🧬 Other Organ Datasets

| Dataset Name | Organ | Task(s) | Modality | Subjects | Labels | Description | Access Link |
|--------------|-------|---------|----------|----------|--------|-------------|-------------|
| **LiTS** | Liver | Tumor Segmentation | CT | 131 | Liver and tumor masks | Liver tumor segmentation challenge dataset. | [Link](https://competitions.codalab.org/competitions/17094) |
| **Pancreas-CT** | Pancreas | Segmentation | CT | 82 | Pancreas masks | Dataset for pancreas segmentation tasks. | [Link](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT) |
| **TotalSegmentator** | Multiple Organs | Segmentation | CT | 1,204 | 104 anatomical structures | Comprehensive dataset for multi-organ segmentation. | [Link](https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/TotalSegmentator.md) |
| **Prostate MR Image Segmentation (PROMISE12)** | Prostate | Segmentation | MRI | 50 | Prostate masks | Dataset for prostate segmentation in MRI. | [Link](https://promise12.grand-challenge.org/) |
| **Retinal Fundus Images (DRIVE)** | Eye | Vessel Segmentation | Fundus Photography | 40 | Vessel annotations | Dataset for blood vessel segmentation in retinal images. | [Link](https://drive.grand-challenge.org/) |

---

## 📚 Additional Resources

- **Awesome-Medical-Dataset**: ^[A curated list of medical datasets and resources.]({"attribution":{"attributableIndex":"6463-2"}}) [GitHub](https://github.com/openmedlab/Awesome-Medical-Dataset)

- **Papers With Code - Medical Datasets**: ^[Collection of medical datasets with associated papers and code.]({"attribution":{"attributableIndex":"6644-1"}}) [Link](https://paperswithcode.com/datasets?mod=medical) [oai_citation:1‡Papers with Code](https://paperswithcode.com/datasets?lang=english&mod=medical&o=newest&page=1&q=&v=lst&utm_source=chatgpt.com)

- **V7 Labs - Healthcare Datasets for Computer Vision**: ^[List of open-source healthcare datasets for computer vision.]({"attribution":{"attributableIndex":"6811-1"}}) [Blog](https://www.v7labs.com/blog/healthcare-datasets-for-computer-vision) [oai_citation:2‡V7 Labs](https://www.v7labs.com/blog/healthcare-datasets-for-computer-vision?utm_source=chatgpt.com)

---

## 🤝 Contributing

^[Contributions are welcome! If you know of a dataset that should be included, please open an issue or submit a pull request with the relevant information.]({"attribution":{"attributableIndex":"7009-1"}})

---

## 📄 License

^[This project is licensed under the MIT License.]({"attribution":{"attributableIndex":"7191-1"}})

---

^[*Note: Always ensure compliance with dataset usage terms and patient privacy regulations when utilizing these datasets.*]({"attribution":{"attributableIndex":"7262-0"}})
