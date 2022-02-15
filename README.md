# AceSum
[EMNLP2021] Aspect-Controllable Opinion Summarization

This PyTorch code was used in the experiments of the research paper

Reinald Kim Amplayo, Stefanos Angelidis, and Mirella Lapata.
[**Aspect-Controllable Opinion Summarization**](https://rktamplayo.github.io/publications/emnlp21.pdf). _EMNLP_, 2021.

The code is cleaned post-acceptance and may run into some errors. Although I did some quick check and saw the code ran fine, please create an issue if you encounter errors and I will try to fix them as soon as possible.

## Data

We used two different datasets from two different papers: Space (Angelidis et al., 2021) and Oposum (Angelidis and Lapata, 2018). We extended the latter to create Oposum+. For convenience we provide the train/dev/test datasets [here](https://drive.google.com/drive/folders/1yofbWkGr5PU474N0dP5aSK1mSTOpiiP6?usp=sharing) which are preprocessed accordingly and saved in separate json or jsonl files. The training corpora are saved with filename `train.jsonl`, while the dev/test sets are named `dev.json`/`test.json` accordingly.

A single line (in jsonl files) or a single example (in json files) is formatted as follows:

```json
{
	"..."
    "reviews": [
    	{
    		"sentences": [
    			"this is the first sentence.",
    			"this is the second one.",
    			"..."
    		],
    		"..."
    	},
       "..."
    ],
    "summary": [
    	"this is the first summary.",
    	"this is the second summary.",
    	"this is the last summary."
    ],
    "..."
}
```

In the example above, `reviews` is a list of reviews represented as a dictionary with `sentences` as keys, and `summary` is a list of reference summaries. There can be other information included in the files but are not used in the code (e.g., review and product identifiers, review rating, etc.). When using the datasets, please also cite the corresponding papers (see below).

## Running the code

AceSum has three steps: induce aspect controllers, create synthetic dataset, and fine-tune T5-small. Below, we show a step-by-step proceedure on how to replicate the results in the paper.

### Step 1: Induce aspect controllers

To do this, we need to train an aspect controller induction model.

We first create a training dataset with weak supervision from seed words (in the `seeds` directory). This can be done by running:

`python src/create_mil_data.py [dataset]`

where `[dataset]` can be `space` or `oposum/[domain]`, where `[domain]` can be any of the six domains of the Oposum dataset. This creates a `train.mil.jsonl` file in the `data/[dataset]` directory. To skip this step, download the corresponding files in the link above.

We then train the aspect controller induction model by running:

`python src/train_mil.py -mode=train -dataset=[dataset] -num_aspects=[num_aspects] -model_name=mil`

where `[num_aspects]` is 6 for Space and 3 for Oposum.

### Step 2: Create the synthetic training dataset

Given the aspect controller model `mil.model`, we (1) create the synthetic training dataset and also (2) preprocess the dev and test splits. These can all be done using the `src/create_sum_data.py` script/

Run the following to create the synthetic training data:

`python src/create_sum_data.py -mode=train -dataset=[dataset] -num_aspects=[num_aspects] -load_model=[mil_model_dir]`

where `[mil_model_dir]` is the location of `mil.model`.

Run the following to create the dev/test *general* and *aspect*-specific summarization datasets:

`python src/create_sum_data.py -mode=eval-[type] -dataset=[dataset] -num_aspects=[num_aspects] -load_model=[mil_model_dir]`

where `[type]` is either `general` or `aspect`.

Additionally for Oposum, we need to combine the domain-specific datasets into one. Use the `combine_oposum.py` script to do this:

`python src/combine_oposum.py [dataset]`.

where `[dataset]` can be one of the following: `train.sum.jsonl`, `dev.sum.general.jsonl`, `dev.sum.aspect.jsonl`, `test.sum.general.jsonl`, `test.sum.aspect.jsonl`.

Again, all of these can be skipped by downloading the files in the link above.

### Step 3: Train the summarization model

This is done by simply running `src/train_sum.py`:

`python src/train_sum.py -mode=train -dataset=[space or oposum] -num_aspects=[6 or 18] -model_name=sum`

Note that the number of aspects of oposum is now `3*6=18` since we have combined the datasets in the previous step. This returns a `sum.model` file after training.

Skip this step by downloading the files in the link above.

### Step 4: Generate the summaries

Generating the summaries can be done by running:

`python src/train_sum.py -mode=eval-[type] -dataset=[space or oposum] -num_aspects=[6 or 18] -load_model=[sum_model_dir]`

where `[type]` is either `general` or `aspect` and `[sum_model_dir]` is the location of `sum.model`.

This will create an output file `acesum.[general or aspect].out` in the corresponding directory. 

If you want to skip everything and just want AceSum, this repo also includes an `output` directory containing AceSum general and aspect-specific summaries. The `i`th line in the file is the summary of the `i`th test example.

## Cite the necessary papers

To cite the paper/code/data splits, please use this BibTeX:

```
@inproceedings{amplayo-etal-2021-aspect,
    title = "Aspect-Controllable Opinion Summarization",
    author = "Amplayo, Reinald Kim  and
      Angelidis, Stefanos  and
      Lapata, Mirella",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.528",
    doi = "10.18653/v1/2021.emnlp-main.528",
    pages = "6578--6593",
}
```

```
@inproceedings{angelidis-lapata-2018-summarizing,
    title = "Summarizing Opinions: Aspect Extraction Meets Sentiment Prediction and They Are Both Weakly Supervised",
    author = "Angelidis, Stefanos  and
      Lapata, Mirella",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    month = oct # "-" # nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D18-1403",
    doi = "10.18653/v1/D18-1403",
    pages = "3675--3686",
}
```

```
@article{angelidis-etal-2021-extractive,
    title = "Extractive Opinion Summarization in Quantized Transformer Spaces",
    author = "Angelidis, Stefanos  and
      Amplayo, Reinald Kim  and
      Suhara, Yoshihiko  and
      Wang, Xiaolan  and
      Lapata, Mirella",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "9",
    year = "2021",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/2021.tacl-1.17",
    doi = "10.1162/tacl_a_00366",
    pages = "277--293",
}
```

If there are any questions, please send me an email: reinald.kim at ed dot ac dot uk
