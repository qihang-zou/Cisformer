import argparse
import os
from pathlib import Path

from cisformer import (
    atac2rna_predict as atac2rna_predict_module,
    # atac2rna_train as atac2rna_train_module,
    # rna2atac_predict as rna2atac_predict_module,
    # rna2atac_train as rna2atac_train_module,
    data_preprocess as data_preprocess_module,
    atac2rna_surround_attention_generate as attention_module,
    generate_default_config as generate_default_config_module
)

def main():
    work_dir = os.getcwd()
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(prog="cisformer", description="Cisformer command-line tools")
    subparsers = parser.add_subparsers(dest="command")

    # generate_default_config
    subparsers.add_parser("generate_default_config", help="Generate cisformer default config")

    # data_preprocess
    preprocess_parser = subparsers.add_parser("data_preprocess", help="Preprocess data")
    preprocess_parser.add_argument("-r", "--rna", required=True, help="Should be .h5ad format")
    preprocess_parser.add_argument("-a", "--atac", required=True, help="Should be .h5ad format")
    preprocess_parser.add_argument("-c", "--config", default=None, help="Config file")
    preprocess_parser.add_argument("-s", "--save_dir", default="./", help="Save directory")
    preprocess_parser.add_argument("--cnt", type=int, default=10000, help="Number of cells per output file")
    preprocess_parser.add_argument("--batch_size", type=int, default=10, help="Batch size")
    preprocess_parser.add_argument("--num_workers", type=int, default=10, help="Number of workers")
    preprocess_parser.add_argument("--atac2rna", action="store_true", help="Process ATAC to RNA")
    preprocess_parser.add_argument("--manually", action="store_true", help="Manual mode")
    preprocess_parser.add_argument("--shuffle", action="store_true", help="Shuffle data")
    preprocess_parser.add_argument("--dec_whole_length", action="store_true", help="Decode whole length")

    # atac2rna_train
    train_a2r = subparsers.add_parser("atac2rna_train", help="Train ATAC to RNA model")
    train_a2r.add_argument("-d", "--data_dir", required=True, help="Data directory")
    train_a2r.add_argument("-n", "--name", required=True, help="Project name")
    train_a2r.add_argument("-s", "--save", default="save", help="Save directory")
    train_a2r.add_argument("-c", "--config_file", default="cisformer_config/atac2rna_config.yaml", help="Config file")
    train_a2r.add_argument("-m", "--model_parameters", default=None, help="Load previous model")

    # atac2rna_predict
    predict_a2r = subparsers.add_parser("atac2rna_predict", help="Predict ATAC to RNA")
    predict_a2r.add_argument("-d", "--data", required=True, help="Data file")
    predict_a2r.add_argument("-m", "--model_parameters", required=True, help="Previous trained model parameters")
    predict_a2r.add_argument("-o", "--output_dir", default="output", help="Output directory")
    predict_a2r.add_argument("-n", "--name", default="cisformer_predicted_rna", help="Output name")
    predict_a2r.add_argument("-c", "--config_file", default="cisformer_config/atac2rna_config.yaml", help="Config file")

    # atac2rna_link
    attention = subparsers.add_parser("atac2rna_link", help="Generate attention matrix")
    attention.add_argument("-d", "--data_path", required=True, help="Data path")
    attention.add_argument("-c", "--celltype_info", required=True, help="Cell type info")
    attention.add_argument("-m", "--model_parameters", required=True, help="Previous trained model parameters")
    attention.add_argument("-o", "--output_dir", default="output", help="Output directory")
    attention.add_argument("-n", "--num_of_cells", type=int, default=None, help="Number of cells")
    attention.add_argument("--config", default="cisformer_config/atac2rna_config.yaml", help="Config file")
    attention.add_argument("--distance", type=int, default=250000, help="Distance threshold")

    # rna2atac_train
    train_r2a = subparsers.add_parser("rna2atac_train", help="Train RNA to ATAC model")
    train_r2a.add_argument("-t", "--train_data_dir", required=True, help="Training data directory")
    train_r2a.add_argument("-v", "--val_data_dir", required=True, help="Validation data directory")
    train_r2a.add_argument("-n", "--name", required=True, help="Project name")
    train_r2a.add_argument("-s", "--save", default="save", help="Save directory")
    train_r2a.add_argument("-c", "--config_file", default="cisformer_config/rna2atac_config.yaml", help="Config file")
    train_r2a.add_argument("-m", "--model_parameters", default=None, help="Load previous model")

    # rna2atac_predict
    predict_r2a = subparsers.add_parser("rna2atac_predict", help="Predict RNA to ATAC")
    predict_r2a.add_argument("-r", "--rna_file", required=True, help="Path of rna adata")
    predict_r2a.add_argument("-m", "--model_parameters", required=True, help="Previous trained model parameters")
    predict_r2a.add_argument("-o", "--output_dir", default="output", help="Load model")
    predict_r2a.add_argument("-n", "--name", default="cisformer_predicted_atac", help="Load model")
    predict_r2a.add_argument("-c", "--config_file", default="cisformer_config/rna2atac_config.yaml", help="Config file")
    predict_r2a.add_argument("--rna_len", default=3600, help="Load model")
    predict_r2a.add_argument("--batch_size", default=2, help="Load model")
    predict_r2a.add_argument("--num_workers", default=2, help="Load model")

    args = parser.parse_args()

    if args.command == "generate_default_config":
        generate_default_config_module.main()

    elif args.command == "data_preprocess":
        data_preprocess_module.main(
            args.rna, args.atac, args.manually, args.atac2rna, args.save_dir,
            args.config, args.batch_size, args.num_workers, args.cnt,
            args.shuffle, args.dec_whole_length
        )

    elif args.command == "atac2rna_train":
        # atac2rna_train_module.main(args.data_dir, args.name, args.save, args.config_file, args.model_parameters)
        os.system(
            f"accelerate launch --config_file cisformer_config/accelerate_config.yaml "
            f"{os.path.join(script_dir, 'atac2rna_train.py')} "
            f"-d {args.data_dir} -n {args.name} -s {args.save} "
            f"-c {args.config_file} -m {args.model_parameters}"
        )

    elif args.command == "atac2rna_predict":
        atac2rna_predict_module.main(args.data, args.output_dir, args.model_parameters, args.config_file, args.name)

    elif args.command == "atac2rna_link":
        attention_module.main(
            args.output_dir, args.data_path, args.celltype_info, args.model_parameters,
            args.num_of_cells, args.config, args.distance
        )

    elif args.command == "rna2atac_train":
        # rna2atac_train_module.main(
        #     args.train_data_dir, args.val_data_dir, args.save, args.name, args.model_parameters, args.config_file
        # )
        os.system(
            f"accelerate launch --config_file cisformer_config/accelerate_config.yaml "
            f"{os.path.join(script_dir, 'rna2atac_train.py')} "
            f"-t {args.train_data_dir} -v {args.val_data_dir} "
            f"-n {args.name} -s {args.save} -c {args.config_file} "
            f"-m {args.model_parameters}"
        )

    elif args.command == "rna2atac_predict":
        # rna2atac_predict_module.main(args.rna_file, args.config_file, args.model_parameters, args.rna_len, args.batch_size, args.num_workers, args.output_dir, args.name)
        os.system(
            f"accelerate launch --config_file cisformer_config/accelerate_config.yaml "
            f"{os.path.join(script_dir, 'rna2atac_predict.py')} "
            f"-r {args.rna_file} -m {args.model_parameters} -o {args.output_dir} "
            f"-n {args.name} -c {args.config_file} "
            f"--rna_len {args.rna_len} --batch_size {args.batch_size} --num_workers {args.num_workers}"
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
