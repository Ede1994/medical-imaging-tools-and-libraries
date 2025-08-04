Medical Imaging Tools and Libraries
A list of modern software, libraries, and resources for medical imaging research, with a focus on the Python ecosystem.

## Table of Contents
Deep Learning Frameworks
Data Handling & Augmentation
Neuroimaging
Radiomics
Visualization & Application Platforms
Core Libraries & Toolkits
File I/O & Formats
General Purpose Python
General Machine Learning
General Image Processing

## Deep Learning Frameworks
Specialized frameworks for building, training, and deploying deep learning models for medical imaging tasks.

### MONAI - Medical Open Network for AI
<img src="https://www.google.com/search?q=https://raw.githubusercontent.com/Project-MONAI/monai.io/main/images/MONAI-logo-color.png" alt="monai" width="50"/>
MONAI is a PyTorch-based, open-source framework for deep learning in healthcare imaging. It provides domain-specific data loading, transformations, models, and metrics to standardize and accelerate research and development.

### nnU-Net - No-New-Net
<img src="https://raw.githubusercontent.com/MIC-DKFZ/nnU-Net/master/documentation/logo/nnU-Net_logo_large.png" alt="nnunet" width="50"/>
nnU-Net is a framework that automatically configures itself, including pre-processing, network architecture, training, and post-processing for any new medical segmentation task. It consistently ranks as a top-performing method for semantic segmentation.

## Data Handling & Augmentation
Tools designed for efficient loading, preprocessing, and augmenting 3D medical images.

### TorchIO
<img src="https://raw.githubusercontent.com/TorchIO-project/torchio/main/docs/_static/logo/logo_light.svg" alt="torchio" width="50"/>
A Python package for efficiently loading, preprocessing, augmenting, and writing 3D medical images in PyTorch. It provides a rich set of intensity and spatial transforms for data augmentation.

## Neuroimaging
A suite of tools specifically designed for the analysis of neuroimaging data (MRI, fMRI, DTI, etc.).

### FastSurfer
<img src="https://raw.githubusercontent.com/Deep-MI/FastSurfer/master/fastsurfer_logo.png" alt="fastsurfer" width="50"/>
A deep-learning based neuroimaging pipeline that provides a fast and accurate alternative to FreeSurfer for volumetric and surface-based analysis. It can perform whole-brain segmentation in under a minute on a GPU.

### ANTs - Advanced Normalization Tools
A state-of-the-art medical image registration and segmentation toolkit. It is widely used for creating image templates and for performing complex, high-dimensional image registration. Relies heavily on ITK.

### FSL - FMRIB Software Library
<img src="https://fsl.fmrib.ox.ac.uk/fsl/wiki_static/fsl/img/fsl-logo-x2.png" alt="fsl" width="50"/>
A comprehensive library of analysis tools for FMRI, MRI and DTI brain imaging data, including tools for image registration, segmentation, and statistical analysis.

### FreeSurfer
<img src="https://surfer.nmr.mgh.harvard.edu/fswiki/images/f/f3/FreeSurfer_logo.png" alt="freesurfer" width="50"/>
An open source software suite for processing and analyzing human brain MRI images. It is famous for its cortical surface reconstruction and thickness analysis capabilities.

### MRtrix3
A set of tools to perform diffusion MRI analyses, from various forms of tractography through to group-level statistical analyses.
SPM - Statistical Parametric Mapping
A software package designed for the analysis of brain imaging data sequences (fMRI, PET, SPECT, EEG, MEG). It focuses on constructing and assessing spatially extended statistical processes to test hypotheses about functional imaging data.

### Nipype
<img src="https://raw.githubusercontent.com/nipy/nipype/master/docs/images/nipype_logo.png" alt="nipype" width="50"/>
A Python project that provides a uniform interface to existing neuroimaging software (like FSL, FreeSurfer, ANTs) and facilitates interaction between these packages within a single, reproducible workflow.

## Radiomics
Tools for extracting a large number of quantitative features from medical images.

### pyradiomics
<img src="https://www.google.com/search?q=https://pyradiomics.readthedocs.io/en/latest/_static/pyradiomics_logo_light.png" alt="pyradiomics" width="50"/>
An open-source Python package for the extraction of Radiomics features from medical imaging data. It is compliant with the Image Biomarker Standardization Initiative (IBSI).

## Visualization & Application Platforms
End-user applications and toolkits for visualization, segmentation, and analysis.

### 3D Slicer
<img src="https://www.slicer.org/w/img_auth.php/1/1f/3DSlicerLogo-H-Color-1273x737.png" alt="slicer" width="50"/>
An open-source software platform for medical image informatics, image processing, and 3D visualization. It's highly extensible through plugins and supports a wide range of functionalities.

### ITK-SNAP
<img src="http://www.itksnap.org/Artwork/snap_logo_2018.png" alt="itksnap" width="50"/>
A software application used to segment structures in 3D medical images. It is particularly known for its powerful semi-automatic segmentation tools using active contours ("snakes").

### VTK - Visualization Toolkit
<img src="https://vtk.org/wp-content/uploads/2021/07/VTK-logo.png" alt="vtk" width="50"/>
An open-source software system for 3D computer graphics, image processing, and visualization. It is the rendering backend for many medical imaging applications, including 3D Slicer.

## Core Libraries & Toolkits
Fundamental C++ libraries with Python wrappers that provide the building blocks for many medical imaging applications.

### ITK - Insight Segmentation and Registration Toolkit
<img src="https://www.google.com/search?q=https://itk.org/itk-logo-with-text.png" alt="itk" width="50"/>
ITK is an open-source, cross-platform system that provides developers with an extensive suite of software tools for image analysis, with a focus on registration and segmentation of multidimensional data.

### SimpleITK
<img src="https://simpleitk.org/SimpleITK.png" alt="simpleitk" width="50"/>
A simplified layer built on top of ITK, intended to facilitate its use in rapid prototyping and education. It provides a procedural, easy-to-use interface, especially popular in Python.

## File I/O & Formats
Libraries for reading and writing common medical imaging file formats.

### DICOM - Digital Imaging and Communications in Medicine
<img src="https://www.dicomstandard.org/assets/images/DICOM-logo-blue-stacked.svg" alt="DICOM" width="50"/>
The international standard to transmit, store, retrieve, print, process, and display medical imaging information. Use libraries like pydicom to work with it in Python.

### NIfTI - Neuroimaging Informatics Technology Initiative
<img src="https://nifti.nimh.nih.gov/nifti-1/documentation/nifti_logo/NIFTI_logo_blue_text.jpg" alt="NIfTI" width="50"/>
A popular file format for neuroimaging, designed to facilitate interoperability between analysis software packages. The .nii or .nii.gz extension is common.

### NiBabel
A Python library providing read/write access to a variety of neuroimaging file formats, including NIfTI, Analyze, and GIFTI.

### dcm2niix
A popular command-line tool designed to convert neuroimaging data from the DICOM format to the NIfTI format.

### NRRD - Nearly Raw Raster Data
<img src="http://teem.sourceforge.net/img/nrrd256.jpg" alt="NRRD" width="50"/>
A file format for N-dimensional raster data, which includes a header and a raw data file. It is commonly used by 3D Slicer and ITK.

## General Purpose Python
The core scientific Python stack, essential for almost any data-driven research.

### NumPy
The fundamental package for numerical computing, providing the ndarray object.

### SciPy library
Provides many user-friendly and efficient numerical routines, such as for numerical integration and optimization.

### Matplotlib
A comprehensive library for creating static, animated, and interactive visualizations.

### Pandas
Provides high-performance, easy-to-use data structures and data analysis tools.

### Seaborn
A data visualization library based on Matplotlib that provides a high-level interface for drawing attractive statistical graphics.

## General Machine Learning
General-purpose ML libraries that are often used as backends for medical imaging frameworks.

### Scikit-learn
<img src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" alt="sklearn" width="50"/>
A free software machine learning library for Python. It features various classification, regression and clustering algorithms and is designed to interoperate with NumPy and SciPy.

### PyTorch
<img src="https://www.google.com/search?q=https://pytorch.org/assets/images/pytorch-logo.png" alt="pytorch" width="50"/>
An open source deep learning platform that provides a seamless path from research prototyping to production deployment. The foundation for MONAI and TorchIO.

### TensorFlow
<img src="https://www.tensorflow.org/images/tf_logo_social.png" alt="tf" width="50"/>
An end-to-end open-source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries, and community resources.

### Keras
<img src="https://www.tensorflow.org/guide/images/keras_logo.png" alt="keras" width="50"/>
Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow. It was developed with a focus on enabling fast experimentation.

## General Image Processing
Classic computer vision and image processing libraries.

### OpenCV - Open Source Computer Vision Library
<img src="https://opencv.org/wp-content/uploads/2023/10/opencv_logo_icon_left_200.png" alt="opencv" width="50"/>
A library of programming functions mainly aimed at real-time computer vision. While it's primarily focused on 2D images, its functions are often useful for pre- or post-processing slices of medical data.

### Scikit-image
<img src="https://scikit-image.org/docs/stable/_static/logo.png" alt="skimage" width="50"/>
A collection of algorithms for image processing. It is built on top of NumPy and is well-integrated with the SciPy stack.

### Pillow
Pillow is the friendly PIL (Python Imaging Library) fork. It is powerful for basic 2D image manipulation and format conversion.

## Community Resources
To further your exploration, this section provides links to other valuable community-curated lists of datasets and tools.

### Dataset Collections
- Awesome-Medical-Dataset: A comprehensive list of public medical datasets for imaging and electronic health records, categorized by anatomy and modality.
- AMID (Awesome Medical Imaging Datasets): A curated list of medical imaging datasets with unified Python interfaces, allowing for easy programmatic access.

### Other Awesome Lists
- awesome-multimodal-in-medical-imaging: A collection of resources, primarily research papers, on multi-modal learning and Large Language Models (LLMs) in medical imaging.

GitHub Topics: Explore repositories tagged with  and  for a broad overview of active projects.

## Archived Projects
This section lists projects that were once influential but are no longer actively maintained.

### NiftyNet
<img src="http://www.niftynet.io/img/niftynet-logo.png" alt="NiftyNet" width="50"/>
Development efforts have been redirected to MONAI, which is the recommended framework for new projects. NiftyNet was a TensorFlow-based open-source platform for medical image analysis.
