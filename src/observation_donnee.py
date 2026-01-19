import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
from data.dataset import create_dataloaders

def main():
    # Paramètres
    CSV_FILE = "./data/raw/prepared_dataset.csv"
    BATCH_SIZE = 32

    # Afficher les lignes du CSV sous forme de DataFrame
    df = pd.read_csv(CSV_FILE)
    print("\nAperçu du DataFrame (10 premières lignes) ---")
    print(df.head(10).to_string(index=False))

    # Créer les dataloaders
    train_loader, val_loader = create_dataloaders(
        csv_file=CSV_FILE,
        batch_size=BATCH_SIZE,
        train_split=0.8,
        img_size=224,
        seed=42
    )
    
    # Afficher les infos
    print(f"\nTrain set: {len(train_loader.dataset)} images")
    print(f"Val set: {len(val_loader.dataset)} images")

if __name__ == "__main__":
    main()