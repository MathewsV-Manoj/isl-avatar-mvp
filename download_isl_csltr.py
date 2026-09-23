"""Download the public ISL-CSLTR sentence corpus to the E: data volume."""

from kaggle.api.kaggle_api_extended import KaggleApi


def main() -> None:
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        "drblack00/isl-csltr-indian-sign-language-dataset",
        path="E:/ISL_Project_Datasets/isl_csltr",
        unzip=False,
        quiet=False,
    )
    print("download complete", flush=True)


if __name__ == "__main__":
    main()
