# Fourth Year Project

## Repository

We have 3 publicly accessible Github repositories.  
The main repository: [https://github.com/JakubCzarlinski/fourth-year-project](https://github.com/JakubCzarlinski/fourth-year-project)  
This repository has 2 repositories as submodules:

- Project Evaluation Code: [https://github.com/NutellaSandwich/4th-Year-Project-Evaluations/tree/main](https://github.com/NutellaSandwich/4th-Year-Project-Evaluations/tree/main)
- Project Dataset: [https://github.com/Alf4ed/fourth-year-project-dataset/tree/main](https://github.com/Alf4ed/fourth-year-project-dataset/tree/main)

### Installation

**Requirements**

- Anaconda installed
- Python 3.12
- A computer with access to a GPU with 24GB RAM or more. An possible option is KUDU on DCS batch compute.
- Github Access or access to Code zip folder.

---

**Installation Process**

1. Clone the repository:
    ```bash
    git clone https://github.com/JakubCzarlinski/fourth-year-project.git
    ```

    If you want access to our submodules such as our Evaluation and dataset repositories run:
    ```bash
    git clone --recurse-submodules https://github.com/JakubCzarlinski/fourth-year-project.git
    ```

2. Navigate to the project directory:
    ```bash
    cd fourth-year-project
    ```

3. Set up a virtual environment:
    ```bash
    conda create --name group python=3.12
    conda activate group
    ```

4. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

If installing on DCS batch compute, we have created `.sbatch` files with everything required loaded to run each of the models. To run these `.sbatch` files, please login into kudu using the following command and then navigate to the project directory:

```bash
ssh kudu
cd path/to/fourth-year-project/
```

**Program Structure**
```
fourth-year-project
├── Models
│   ├── JRAP
│   └── experiments
│       ├── DDD
│       ├── DDD_fast
│       ├── DiffusionGuard
│       ├── photoguard
│       └── watermark_attacker
├── fourth-year-project-dataset
└── 4th-Year-Project-Evaluation
```

## JRAP Model Execution Guide

The **JRAP** model is designed to create JPEG-resistant adversarial perturbations for disrupting Deepfake models. To execute this model, ensure that you are running it on a machine with **24GB or more GPU memory**.

### Running JRAP on DCS Batch Compute

To run JRAP, use the `jrap.sbatch` script. This script performs the following tasks automatically:

- Loads the required **CUDA module** on DCS.
- Activates the **Conda environment** created for this project. (You may need to change `conda activate group` to the name of your Conda environment.)
- Submits the **Python script job** to Slurm for running on **KUDU**.

Make sure to run the model on either the **falcon** or **gecko** partitions, as they meet the recommended hardware requirements.

#### To execute the script:

```bash
ssh kudu
cd path/to/fourth-year-project/Models/JRAP/
sbatch jrap.sbatch
```

Before running the script, ensure that the input images follow the correct structure: one for original images, one for masks, and one for prompts. Make sure the images are of size 512x512. The expected structure should look like this:
```
example_folder  
├── original  
│   └── filename.png  
├── masks  
│   └── filename_masked.png  
└── prompts  
    └── filename_prompts.txt
```

The example_folder and the filename can be changed by modifying the configuration in attack.py in the src/ folder. Given this configuration dictionary in the file, we can adapt it to allow for any filename and folder as long as it follows the structure shown above.

```python
jrap_args = {
    "image_size": 512,
    "image_size_2d": (512, 512),
    "image_folder": "./example_folder/",
    "image_filenames": ["filename"],
    "num_inference_steps": 4,
    "evaluation_metric": "COS_NORMED",
    "t_schedule": [720],
    "t_schedule_bound": 10,
    "centroids_n_samples": 50,
    "loss_depth": [4096, 1024, 256, 64],
    "iters": 268,
    "grad_reps": 7,
    "loss_mask": True,
    "eps": 13,
    "step_size": 3.0,
    "pixel_loss": 0
}
```

Change the values of image_folder and image_filenames based on the user's given images. Make sure this folder exists in the top level of the JRAP folder.

#### Output Directory

The outputs for this model will be provided in the Images folder in the JRAP folder. Make sure the Images folder exists. If it doesn't, please create both the sbatch and Images folders using the following commands:

```bash
mkdir sbatch
mkdir Images
```