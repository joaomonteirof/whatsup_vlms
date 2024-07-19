import os
import functools
import argparse
from collections import defaultdict
from multiprocessing import Pool
from dataset_zoo import get_dataset
from misc import seed_all
import datasets


def convert_to_hf_dataset(dataset):
    data_rows = defaultdict(list)
    for row in dataset:
        for k, v in row.items():
            data_rows[k].append(v)

    hf_dataset = datasets.Dataset.from_dict(data_rows)

    return hf_dataset


def get_and_prepare_hf_dataset(dataset_name, output_dir):
    raw_dataset = get_dataset(dataset_name, download=True)
    hf_converted_dataset = convert_to_hf_dataset(raw_dataset)
    output_path = os.path.join(args.output_dir, dataset_name)
    hf_converted_dataset.save_to_disk(output_path)

    return {dataset_name: hf_converted_dataset}


def join_save_and_push_datasets(hf_hub_path, dataset_names, output_dir):

    if hf_hub_path is None:
        return

    data_splits = {}

    for dataset_name in dataset_names:
        data_path = os.path.join(output_dir, dataset_name)
        hf_dataset = datasets.load_from_disk(data_path)
        data_splits[dataset_name] = hf_dataset

    complete_dataset = datasets.Dataset.from_dict(data_splits)
    complete_data_path = os.path.join(output_dir, "hf_full_dataset")

    if not os.path.exists(complete_data_path):
        complete_dataset.save_to_disk(complete_data_path)
        complete_dataset.push_to_hub(args.hf_hub_path)


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_hub_path",
        type=str,
        default=None,
        help="Will pushthe data to the huggingface hub using the provided path if not None. (Default: None)",
    )
    parser.add_argument("--output_dir", default="./data", type=str)
    parser.add_argument("--n_processes", default=1, type=int)
    parser.add_argument("--seed", default=1, type=int)
    return parser.parse_args()


def main(args):
    seed_all(args.seed)

    datasets_to_prepare = [
        "VG_Relation",
        "VG_Attribution",
        "COCO_Order",
        "Controlled_Images_A",
        "Controlled_Images_B",
        "COCO_QA_one_obj",
        "COCO_QA_two_obj",
        "VG_QA_one_obj",
        "VG_QA_two_obj",
        # "Flickr30k_Order",
    ]

    # Remove datasets already in disk
    filtered_dataset_list = [
        dataset_name
        for dataset_name in datasets_to_prepare
        if not os.path.exists(os.path.join(args.output_dir, dataset_name))
    ]

    print(filtered_dataset_list)

    # Run data preparations in parallel and save results to disk
    with Pool(processes=args.n_processes) as P:
        P.map(
            functools.partial(get_and_prepare_hf_dataset, output_dir=args.output_dir),
            filtered_dataset_list,
        )

    # Load, join, and save/push the complete dataset
    join_save_and_push_datasets(
        args.hf_hub_path,
        datasets_to_prepare,
        args.output_dir,
    )


if __name__ == "__main__":
    args = config()
    main(args)
