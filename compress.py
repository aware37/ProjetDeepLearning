from PIL import Image, ImageOps
import os
import shutil

# ====== PARAMÈTRES ======
INPUT_DIR = "Data_Projet_Complet"      # dossier source
OUTPUT_DIR = "data_projet_compressed"   # dossier destination
MAX_SIZE = 1024                         # taille max du côté le plus long
JPEG_QUALITY = 85                       # 85–90 recommandé

# =======================
print(f"Compression de {INPUT_DIR} vers {OUTPUT_DIR}...")

total_images = 0
errors = 0
copied = 0

# Parcourir récursivement tous les fichiers JPG
for root, dirs, files in os.walk(INPUT_DIR):
    for filename in files:
        if filename.lower().endswith((".jpg", ".jpeg")):
            input_path = os.path.join(root, filename)
            
            # Recréer la structure du dossier dans OUTPUT_DIR
            relative_path = os.path.relpath(root, INPUT_DIR)
            output_root = os.path.join(OUTPUT_DIR, relative_path)
            os.makedirs(output_root, exist_ok=True)
            
            output_path = os.path.join(output_root, filename)
            
            try:
                with Image.open(input_path) as img:
                    # Corrige l'orientation EXIF
                    img = ImageOps.exif_transpose(img)
                    
                    # Redimensionne en gardant le ratio
                    img.thumbnail((MAX_SIZE, MAX_SIZE))
                    
                    # Force RGB
                    img = img.convert("RGB")
                    
                    # Sauvegarde compressée
                    img.save(
                        output_path,
                        format="JPEG",
                        quality=JPEG_QUALITY,
                        optimize=True
                    )
                
                total_images += 1
                if total_images % 100 == 0:
                    print(f"{total_images} images compressées...")
                    
            except Exception as e:
                # Si erreur : copier le fichier tel quel
                try:
                    shutil.copy(input_path, output_path)
                    copied += 1
                    print(f"{filename} copié tel quel (erreur: {str(e)[:50]})")
                except Exception as copy_error:
                    errors += 1
                    print(f"Impossible de traiter {filename}: {copy_error}")

print(f"\nTerminé !")
print(f"{total_images} images compressées")
print(f"{copied} images copiées telles quelles")
print(f"{errors} erreurs")
print(f"Dossier : {OUTPUT_DIR}")