# Lymphocyte Analysis in IHC stained histopathological images
</b>

## Project Overview
</b>

Presence of Tumor Infitrating Lymphocytes in cancer patients is related to patient prognosis after undergoing surgery or immunotherapy Therefore, lymphocyte detection and quantification is very important in cancer diagnosis. In this work, a lymphocyte analysis framework is developed that localizes lymphocytes undergoing two phases. The framework consists of a screening phase followed by a localization phase. For each phase a Deep CNN based model is implemented. 

<img src="images/Overall_workflow.png" > 
</b>
Figure 1: Overview of the proposed lymphocyte analysis workflow.

## Histopathological Lymphcoyte Dataset
</b>
Two publically available IHC stained lymphocytic datasets are used. For classification/screening phase we used "LYSTO dataset" and for detection/localization phase we employed dataset released by the "NuClick developers". Both datasets contained lymphocyte patch images stained using immunohistochemistry staining, which stains lymphocytes in brown color rings surronding a blue nucleus. Patch images were obtained from pathologist marked ROIs obtaned from different hospitals in Netherland. These ROI images are categorized into three regions a) artifact regions (containing some staining issues) b) regular regions (with sparse lymphocytes) and c) dense regions (with overlapping lymphocytes).

## Screening Phase
</b>

Deep CNN based classification is performed to select candiate lymphocytic patches. In this regard a custom Lymph-DilNet is proposed. 

<img src="images/Screening_model.png" > 
</b>
Figure 2: Architectural diagram of the screening model.

## Localization Phase
</b>

Once the candidate lymphoctyic patches are selected, they undergo a localization phase, where each lymphocyte in an image is localized. In this regard, we modified the backbone architecture of the state-of-the-art instance segmentation model (MaskRCNN). 

<img src="images/Localization_model.png" > 
</b>
Figure 3: Architectural diagram of the localization model.

## Requirements
</b>

</b> Python = 3.8

</b> Cuda = 11.4

</b> Torch = 1.12.0

</b> Torchvision = 0.13.0

</b> Detectron2


## Implementation Details

Trained models are in trained_models directory.

Inference and model definition files are in py_files directory. Load the desired model by setting its name in inference file.

## More Information

For more information email at the provided email address. 
Email : zunirauf01@gmail.com

