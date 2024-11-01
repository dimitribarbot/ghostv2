import os
from typing import cast, Optional

from simple_parsing import ArgumentParser
from tqdm import tqdm

import cv2

from utils.preprocessing.convert_arguments import ConvertArguments


def get_save_path(root_folder: str, source_image: str, output_folder: str, output_extension: str):
    image_name = os.path.splitext(os.path.basename(source_image))[0]
    relative_path = os.path.relpath(os.path.dirname(source_image), root_folder)
    save_folder = os.path.join(output_folder, relative_path)
    save_path = os.path.join(save_folder, f"{image_name}{output_extension}")
    return save_path


def process_one_image(
    source_image: str,
    save_path: str,
):
    image = cv2.imread(source_image, cv2.IMREAD_COLOR)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, image)
    return save_path


def process(
    source_image: Optional[str],
    source_folder: Optional[str],
    output_folder: str,
    output_extension: str,
    overwrite: bool,
):
    if source_image is None and source_folder is None:
        raise ValueError("Arguments 'source_image' and 'source_folder' cannot be both empty.")

    if source_folder is None:
        if not os.path.exists(source_image):
            raise ValueError(f"Arguments 'source_image' {source_image} points to a file that does not exist.")
        
        root_folder = os.path.dirname(source_image)
        save_path = get_save_path(root_folder, source_image, output_folder, output_extension)
        if overwrite or not os.path.exists(save_path):
            print(f"Processing image {source_image}.")
            save_path = process_one_image(
                source_image,
                save_path,
            )
        else:
            print(f"Not processing image {source_image} as target {save_path} already exists and overwrite is false.")
    else:
        if not os.path.exists(source_folder):
            raise ValueError(f"Arguments 'source_folder' {source_folder} points to a folder that does not exist.")
    
        print(f"Processing images in folder {source_folder}.")
        print("Counting number of files to process.")
        total = sum([len(list(filter(lambda file: overwrite or not os.path.exists(get_save_path(source_folder, os.path.join(root, file), output_folder, output_extension)), files))) \
                     for root, _, files in os.walk(source_folder)])
        print(f"Number of files to process: {total}.")
        with tqdm(total=total) as pbar:
            for root, _, files in os.walk(source_folder):
                for file in files:
                    source_image = os.path.join(root, file)
                    save_path = get_save_path(source_folder, source_image, output_folder, output_extension)
                    if overwrite or not os.path.exists(save_path):
                        save_path = process_one_image(
                            source_image,
                            save_path,
                        )
                        pbar.update()


def main(args: ConvertArguments):
    process(
        args.source_image,
        args.source_folder,
        args.output_folder,
        args.output_extension,
        args.overwrite,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(ConvertArguments, dest="arguments")
    args = cast(ConvertArguments, parser.parse_args().arguments)
    
    main(args)