# ChangeChip
<img align="right" width="170" src="assets/CC_logo.png">

*ChangeChip* was developed to detect changes between an inspected PCB image and a reference (golden) PCB image, in order to detect defects in the inspected PCB.\
The system is based on Image Processing, Computer Vision and Unsupervised Machine Learning.\
*ChangeChip* is targeted to handle optical images, and also radiographic images, and may be applicable to other technologies as well.\
We note that *ChangeChip* is not limited to PCBs only, and may be suitable to other systems that require object comparison by their images.
The workflow of *ChangeChip* is presented as follows:

<img align="center" width="250" height="" src="assets/workflow.PNG">

## Requirements:
- Create a conda environment with python 3.11
```
conda create --name changechip python=3.11
conda activate changechip
```

- install requirements
```
pip install -r requirements.txt
```

```
## Running:

not implemented yet

# CD-PCB
As part of this work, a small dataset of 20 pairs of PCBs images was created, with annotated changes between them. This dataset is proposed for evaluation of change detection algorithms in the PCB Inspection field. The dataset is available [here](https://drive.google.com/file/d/1b1GFuKS88nKaH-Nfx2XmlhwulUxMwwBA/view?usp=sharing).

---

#### Example of pairs from CD-PCB, the ground truth changes and *ChangeChip* results according to the parameters described in the Results section in the paper. 
#### The red circles are for easy identification by the reader.

<img align="center" src="assets/cd_pcb_results_a.jpg">
<img align="center" src="assets/cd_pcb_results_b.jpg">
