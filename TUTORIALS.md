# Hands-on Tutorials


## 1. Preparing the Python Environment with `uv`

`uv` is a fast Python package installer and resolver. Follow these steps to set up your environment:

1.  **Install `uv` (if you haven't already)**:

    ```bash
    # For macOS and Linux (using curl)
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

    ```powershell
    # For Windows (using PowerShell)
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```



2.  **Clone the project, navigate to its directory, and initialize the environment**:

    Clone the project repository (replace `[YOUR_REPOSITORY_URL]` with the actual URL), navigate into the project directory (assuming it creates a folder named `tutorial_model_benchmark`), and then run `uv sync` to install all dependencies:

    ```bash
    git clone [YOUR_REPOSITORY_URL]
    cd tutorial_model_benchmark
    uv sync
    ```


## 2. Downloading Video Assets via CLI Usage

The `video_download.py` script allows you to list available video assets or download specific ones using command-line arguments.

### 2.1. Listing Available Assets

To see all the video assets you can download, use the `--show` argument:

```bash
uv run video_download.py --show
```

This command will output a list of all available video asset names, such as `VEHICLES`, `PEOPLE_WALKING`, `MILK_BOTTLING_PLANT`, etc.

### 2.2. Downloading a Specific Asset

To download a particular video asset, use the `--video` argument followed by the asset's name (e.g., `PEOPLE_WALKING`):

```bash
uv run video_download.py --video PEOPLE_WALKING
```

The script will download the `people-walking.mp4` file to your current directory.

### 2.3. Downloading All Assets

If you want to download all available video assets, use the `--video all` argument:

```bash
uv run video_download.py --video all
```

This will iterate through all assets and download them one by one.