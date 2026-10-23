# Dataset

This project uses the **Phishing Websites** dataset by Rami Mohammad, Fadi Thabtah and Lee McCluskey (UCI Machine Learning Repository, 2012, https://doi.org/10.24432/C51W2X). It has 11,055 websites described by 30 features with values -1, 0 or 1, and a `Result` column where **-1 = phishing** and **1 = legitimate**.

The file is not included in this repository. Get it from either source and place it in this folder:

1. UCI: https://archive.ics.uci.edu/dataset/327/phishing+websites (the download contains `Training Dataset.arff`; the code reads `.arff` directly).
2. Kaggle: search "phishing websites dataset" (for example `mmelgizin/phishing-websites-dataset`) and download the CSV.

Then run `python scripts/run_all.py`. The loader picks up the first `.csv` or `.arff` it finds here, or you can pass `--data PATH`.

Whichever copy you use, the loader checks the label direction: if valid SSL (`SSLfinal_State = 1`) correlates *positively* with the phishing label, the labels are inverted and it stops with an error.
