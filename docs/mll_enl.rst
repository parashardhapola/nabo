=========================
MLL-ENL tutorial workflow
=========================

You may download the raw dataset along with all the notebooks (approx 43 MB) for this example using following link: Download_

.. _Download: https://files.osf.io/v1/resources/7t3rc/providers/osfstorage/5b452fcd4e95590012affb23?action=download&version=1&view_only=338f2b3896fe48ab99f52cc54cd01653&direct
 
Extract the downloaded `mll_enl.zip` file to any folder. Once extracted, you will find the following directory structure:

::

	.
	├── analysis_data
	├── notebooks
	│   ├── 1_preprocessing.ipynb
	│   ├── 2_mapping.ipynb
	│   ├── 3_markers.ipynb
	│   └── 4_control_mappings.ipynb
	└── raw_data
		├── MLL_ENL
		│   ├── barcodes.tsv
		│   ├── genes.tsv
		│   └── matrix.mtx
		└── WT
		├── barcodes.tsv
		├── genes.tsv
		└── matrix.mtx

Files with .ipynb extension are notebooks file. If you do not have a jupyter notebook server running, then open your terminal and type `jupyter notebook`. Your browser should open a new window automatically, navigate to the notebook location and launch the notebooks. Start with '1_preprocessing.ipynb' or continue from where you last left. Please run '1_preprocessing' notebook first, followed by '2_mapping' and thereafter, the rest two notebooks can be run in any order.

Read static notebooks here:

.. toctree::
	:maxdepth: 1

	analysis/mll_enl/notebooks/1_preprocessing.ipynb
	analysis/mll_enl/notebooks/2_mapping.ipynb
	analysis/mll_enl/notebooks/3_markers.ipynb
	analysis/mll_enl/notebooks/4_control_mappings.ipynb

