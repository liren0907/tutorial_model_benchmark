import argparse
from supervision.assets import download_assets, VideoAssets


def main():
    parser = argparse.ArgumentParser(description="Download video assets.")
    parser.add_argument(
        "--video",
        type=str,
        choices=[asset.name for asset in VideoAssets] + ["all"],
        required=False,
        help="Select the video asset to download (e.g., VEHICLES, MILK_BOTTLING_PLANT) or 'all' to download all assets"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="List all available video assets and exit"
    )
    args = parser.parse_args()

    if args.show:
        print("Available Video Assets:")
        for asset in VideoAssets:
            print(f"- {asset.name}")
        exit()

    if args.video:
        if args.video == "all":
            for asset in VideoAssets:
                download_assets(asset)
        else:
            try:
                selected_asset = VideoAssets[args.video]
                download_assets(selected_asset)
            except KeyError:
                print(f"Error: Invalid asset name '{args.video}'. Please choose from {', '.join([asset.name for asset in VideoAssets])} or 'all'")


if __name__ == "__main__":
    main()
