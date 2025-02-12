# Datasets
We intend to use the following 4 billion point datasets and their sampled down variants in the experiments:
1. [BIGANN](http://corpus-texmex.irisa.fr/) consists of SIFT descriptors applied to images extracted from a large image dataset.
2. [SpaceV1B](https://github.com/microsoft/SPTAG/tree/main/datasets/SPACEV1B) is a new web search-related dataset released by Microsoft Bing for this competition. It consists of document and query vectors encoded by the Microsoft SpaceV Superior model to capture generic intent representation.
3. [Yandex DEEP-1B](https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search) image descriptor dataset consisting of the projected and normalized outputs from the last fully connected layer of the GoogLeNet model, which was trained on the Imagenet classification task.
4. [Yandex Text-to-Image-1B](https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search) is a new cross-model dataset (text and visual), where database and query vectors have different distributions in a shared representation space. The base set consists of Image embeddings produced by the Se-ResNext-101 model, and queries are textual embeddings produced by a variant of the DSSM model. Since the distributions are different, a 50M sample of the query distribution is provided.

All datasets are in the common binary format that starts with 8 bytes of data consisting of num_points(uint32_t) num_dimensions(uint32) followed by num_pts X num_dimensions x sizeof(type) bytes of data stored one vector after another. Data files will have suffixes .fbin, .u8bin, and .i8bin to represent float32, uint8 and int8 type data. Note that a different query set will be used for evaluation. The details of the datasets along with links to the base, query and sample sets, and the ground truth nearest neighbors of the query set are listed below.

The ground truth binary files for k-NN search consist of the following information: num_queries(uint32_t) K-NN(uint32) followed by num_queries X K x sizeof(uint32_t) bytes of data representing the IDs of the K-nearest neighbors of the queries, followed by num_queries X K x sizeof(float) bytes of data representing the distances to the corresponding points. The distances help identify neighbors tied in terms of distances. In recall calculation, returning a neighbor not in the ground truth set but whose distance is tied with an entry in the ground truth is counted as success.

| Dataset             | Datatype | Dimensions   |  Distance   | Base data | Sample data | Query data | Size  | Release Terms                                                                    | 
| :----------------:  | :------: | :----:       |   :----:    | :----:    |  :----:     |  :----:    |     :----:    |   :----:                                                                         |
| BIGANN (SIFT)       | uint8    |  128         |     L2      |   1B      |  100M       | 10K        |   132GB    | [CC0](https://creativecommons.org/public-domain/cc0/)                           |
| Microsoft SPACEV    | int8     |  100         |     L2      |   1B      |  100M       | 29.3K      |   131GB  | [O-UDA](https://github.com/microsoft/SPTAG/blob/main/datasets/SPACEV1B/LICENSE) |
| Yandex DEEP         | float32  |  96          |     L2      |   1B      |  350M       | 10K        |    483GB   | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.en)                |
| Yandex Text-to-Image| float32  |  200         |inner-product|   1B      |   50M       | 100K       |    783GB   | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.en)                |

# Dependencies
```bash
pip install aria2c
```

# Download 1B Datasets

```bash
./download.sh
```

# Download 1M, 10M and 100M Datasets

```bash
./download_prep.sh 
```

# File Structure
```
datasets/                 # datasets used in experiment created by the download.sh script
                          # (Tip: You can create a symbolic link for this directory before you run the download script to change the location of the downloaded file)
    bigann/               # BIGANN / SIFT
    spacev/               # Microsoft SPACEV
    deep/                 # Yandex DEEP
    text2image/        # Yandex Text-to-Image