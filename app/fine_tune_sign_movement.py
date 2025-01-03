import argparse
from sign_movement_model import fine_tune_movement

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, default="data/training_data_movements.json")
    parser.add_argument("--model-save-dir", type=str, default="saved_movement_model")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    fine_tune_movement(
        train_file=args.data_file,
        model_save_dir=args.model_save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length
    ) 