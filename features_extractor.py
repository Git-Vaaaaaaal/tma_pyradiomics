import pandas as pd
from pathlib import Path
import SimpleITK as sitk
from radiomics import featureextractor, setVerbosity
import logging
import numpy as np
import os

setVerbosity(logging.ERROR)

def extract_radiomics(image_dir, mask_dir, output_csv):
    extractor = featureextractor.RadiomicsFeatureExtractor("temp_params.yaml")
    rows = []

    for img_path in sorted(Path(image_dir).glob("*.tiff")):
        print(f"Processing {img_path.name}...")
        mask_path = Path(mask_dir) / f"{img_path.stem}.tiff"
        if not mask_path.exists():
            print(f"[SKIP] Masque manquant pour {img_path.name}")
            continue

        # --- Lecture image ---
        image = sitk.ReadImage(str(img_path))
        if image.GetNumberOfComponentsPerPixel() > 1:
            image = sitk.VectorIndexSelectionCast(image, 0)
        image = sitk.Cast(image, sitk.sitkFloat32)

        # --- Lecture masque ---
        mask = sitk.ReadImage(str(mask_path))
        if mask.GetNumberOfComponentsPerPixel() > 1:
            mask = sitk.VectorIndexSelectionCast(mask, 0)

        # Cast en float32 AVANT le threshold pour éviter les surprises de type
        mask = sitk.Cast(mask, sitk.sitkFloat32)

        # Binarisation : tout pixel > 0 devient 1
        mask = sitk.BinaryThreshold(mask, lowerThreshold=1, upperThreshold=255,
                                    insideValue=1, outsideValue=0)
        mask = sitk.Cast(mask, sitk.sitkUInt8)

        # Vérification
        vals = np.unique(sitk.GetArrayFromImage(mask))
        print(f"{img_path.name} → valeurs masque après threshold : {vals}")
        if 1 not in vals:
            print(f"  [SKIP] Masque vide pour {img_path.name}")
            continue

        # --- Correction du spacing ---
        # Si l'image TIFF n'a pas de métadonnées spatiales, son spacing est [1,1]
        # On force le masque à avoir le même spacing/origine que l'image
        mask.SetSpacing(image.GetSpacing())
        mask.SetOrigin(image.GetOrigin())
        mask.SetDirection(image.GetDirection())

        # --- Resample masque sur la grille image (au cas où tailles différentes) ---
        if mask.GetSize() != image.GetSize():
            print(f"  Resample masque {mask.GetSize()} → {image.GetSize()}")
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(image)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetDefaultPixelValue(0)
            resampler.SetOutputPixelType(sitk.sitkUInt8)
            mask = resampler.Execute(mask)

        # --- Extraction ---
        try:
            feats = extractor.execute(image, mask, label=1)
            rows.append({
                "image": img_path.name,
                **{k: v for k, v in feats.items() if not k.startswith("diagnostics_")}
            })
        except Exception as e:
            print(f"  [ERREUR] {img_path.name} : {e}")
            continue

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    return df

marker_list = ["BCL2"] #, "BCL6", "CD10", "HE", "MUM1", "MYC"

for marker in marker_list :
    df = extract_radiomics(
        image_dir  = os.path.join("./img_dir", marker),
        mask_dir   = os.path.join("./mask_dir", marker),
        output_csv = f"{marker}_features.csv"
    )