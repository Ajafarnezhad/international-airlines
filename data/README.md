# Dataset

This project analyzes the **International Airlines Traffic by City Pairs**
dataset published on Kaggle by imtkaggleteam:

> https://www.kaggle.com/datasets/imtkaggleteam/international-airlines-traffic-by-city-pairs

It contains 89k+ rows of monthly passenger, freight, and mail traffic
between Australian and foreign ports, drawn from Australia's Bureau of
Infrastructure and Transport Research Economics (BITRE) statistics.

The dataset is not bundled in this repository. To use it:

```bash
kaggle datasets download -d imtkaggleteam/international-airlines-traffic-by-city-pairs -p data --unzip
```

This produces `data/city_pairs.csv`, which every command in this project
expects by default (`--data data/city_pairs.csv`).
