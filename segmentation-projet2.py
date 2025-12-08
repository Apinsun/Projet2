import os
import requests
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from tqdm import tqdm
import base64
import io
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import time
from sklearn.metrics import confusion_matrix

def get_image_dimensions(img_path):
    """
    Get the dimensions of an image.
    """
    original_image = Image.open(img_path)
    return original_image.size

def decode_mask_bytes(mask_input, width, height):
    """
    Prépare le masque (qu'il soit en bytes ou déjà en objet PIL) 
    et le convertit en tableau NumPy redimensionné.
    """
    # 1. Vérification du type d'entrée
    # Si c'est déjà une image PIL (cas de InferenceClient), on l'utilise directement
    if isinstance(mask_input, Image.Image):
        mask_image = mask_input
    # Si ce sont des octets (cas anciens ou autres API), on décode
    else:
        mask_image = Image.open(io.BytesIO(mask_input))
    
    # 2. Redimensionnement vers la taille de l'image originale
    # On utilise NEAREST pour ne pas interpoler les classes (garder des entiers stricts)
    mask_image = mask_image.resize((width, height), Image.NEAREST)
    
    # 3. Conversion en NumPy
    mask_array = np.array(mask_image)
    
    # 4. Si l'image a plusieurs canaux (ex: RGB), on ne garde que le premier
    # car un masque de segmentation est une map 2D
    if mask_array.ndim == 3:
        mask_array = mask_array[:, :, 0]
        
    return mask_array

def create_masks(results, width, height):
    """
    Combine multiple class masks into a single segmentation mask.
    """
    combined_mask = np.zeros((height, width), dtype=np.uint8) 

    # Process non-Background masks first
    for result in results:
        label = result['label']
        class_id = CLASS_MAPPING.get(label, 0)
        if class_id == 0:  # Skip Background initially
            continue
            
        # Appel de la fonction corrigée qui gère les deux types (PIL ou Bytes)
        mask_array = decode_mask_bytes(result['mask'], width, height)
        
        # On applique le masque
        combined_mask[mask_array > 0] = class_id

    # Process Background last (optional logic depending on preference)
    for result in results:
        if result['label'] == 'Background':
            mask_array = decode_mask_bytes(result['mask'], width, height)
            # On met à 0 (Background) seulement là où le masque dit "C'est du fond"
            # Note : Selon la logique, on peut vouloir ne pas écraser les objets détectés avant.
            # Ici, on écrase avec 0.
            combined_mask[mask_array > 0] = 0 

    return combined_mask

def segment_images_batch(list_of_image_paths):
    """
    Segmente une liste d'images en utilisant l'API Hugging Face.

    Args:
        list_of_image_paths (list): Liste des chemins vers les images.

    Returns:
        list: Liste des masques de segmentation (tableaux NumPy).
              Contient None si une image n'a pas pu être traitée.
    """
    batch_segmentations = []


    # Maintenant, on utilise l'API huggingface
    # ainsi que les fonctions données plus haut pour ségmenter nos images.
    imageType = "image/png"
    headers["Content-Type"] = imageType



    client = InferenceClient(
    provider="hf-inference",
    api_key=api_token,
    )
    for ip in tqdm(list_of_image_paths) :
        try:
    # Lire l'image en binaire
    # Et mettez le contenu de l'image dans la variable image_data
            image_path =  os.path.join(image_dir, ip)
            image_w, image_h = get_image_dimensions(image_path)
            response = client.image_segmentation(image_path, model="sayeed99/segformer_b3_clothes")
            segmentation_mask = create_masks(response, image_w, image_h)
            batch_segmentations.append(segmentation_mask)
            time.sleep(0.1)
        except Exception as e:
            print(f"Une erreur est survenue : {e}")

    return batch_segmentations

def display_segmented_images_batch(original_image_paths, segmentation_masks):
    """
    Affiche les images originales et leurs masques segmentés.

    Args:
        original_image_paths (list): Liste des chemins des images originales.
        segmentation_masks (list): Liste des masques segmentés (NumPy arrays).
    """
    # Matplotlib, ça vous parle ?
    # Alors... au travail ! 😉
    """
    Affiche les images originales et leurs masques segmentés côte à côte.
    """
   # On prépare le dictionnaire inverse (ID -> Nom) une seule fois au début
    # (Supposant que class_mapping est du type {'Fond': 0, 'Chapeau': 1})
    id_to_label = {v: k for k, v in CLASS_MAPPING.items()}
    with PdfPages('rapport_segmentation.pdf') as pdf:
        
        # Boucle principale
        for i, (nfp, segmentation_mask) in enumerate(zip(original_image_paths, segmentation_masks)):
            
            # CORRECTION 1 : On veut 3 colonnes (Originale | Pred | Vérité)
            # figsize agrandi pour accommoder 3 images
            fig, axes = plt.subplots(1, 3, figsize=(20, 8)) 
            
            # --- 1. Image Originale (Gauche) ---
            try:
                # Utilisation de os.path.join pour être robuste
                img_path = os.path.join(image_dir, nfp)
                original_image = Image.open(img_path)
                axes[0].imshow(original_image)
                axes[0].set_title(f"Source : {os.path.basename(nfp)}")
            except Exception as e:
                axes[0].text(0.5, 0.5, "Image introuvable", ha='center')
                print(f"Erreur image {nfp}: {e}")
            axes[0].axis('off') 
            
            # --- Notre segmentation au milieu ---
            unique_classes = np.unique(segmentation_mask)
            
            # On stocke l'objet 'im' pour récupérer les couleurs pour la légende
            im = axes[1].imshow(segmentation_mask, cmap='tab20', interpolation='nearest', vmin=0, vmax=19)
            axes[1].set_title("Prédiction IA")
            axes[1].axis('off')
            
            # --- 3. Vérité Terrain (Droite)  ---
            mask_filename = mask_filename = nfp.replace("image_", "mask_")
            mask_path = os.path.join(image_dir_masks, mask_filename) 
            
            try:
                true_mask_pil = Image.open(mask_path)
                
                # Redimensionnement (Nearest Neighbor) si les tailles diffèrent
                # PIL utilise (largeur, hauteur), alors que numpy shape est (hauteur, largeur)
                if true_mask_pil.size != (segmentation_mask.shape[1], segmentation_mask.shape[0]):
                    true_mask_pil = true_mask_pil.resize((segmentation_mask.shape[1], segmentation_mask.shape[0]), resample=Image.NEAREST)
                
                # Conversion en tableau numpy
                true_mask = np.array(true_mask_pil)
                
                axes[2].imshow(true_mask, cmap='tab20', interpolation='nearest', vmin=0, vmax=19)
                axes[2].set_title(f"Vérité Terrain ({mask_filename})")
                
            except FileNotFoundError:
                axes[2].text(0.5, 0.5, "Masque non trouvé", ha='center')
                axes[2].set_title("Vérité Terrain (Manquante)")
                print(f"Masque de vérité terrain non trouvé pour l'image {nfp} : {mask_path}")
            
            axes[2].axis('off')

            # calcul de la mean intersection over union
            acc, miou , _ = calculer_metrics_segmentation(segmentation_mask, true_mask)
                
            # On prépare le texte à afficher
            score_text = f"Global Accuracy: {acc:.2%}  |  Mean IoU (mIoU): {miou:.2%}"
            
            # Petit bonus couleur : Vert si bon score, Rouge si mauvais
            text_color = "darkgreen" if miou > 0.6 else "firebrick"

            # --- Légende (Dynamique basée sur la prédiction) ---
            legend_patches = []
            
            for idx, val in enumerate(unique_classes):
                class_id = int(val) # On force en entier Python simple (sécurité anti-bug)
                
                if im.norm(class_id) is not None:
                    color = im.cmap(im.norm(class_id))
                    label_name = id_to_label.get(class_id, f"Class {class_id}")
                    patch = mpatches.Patch(color=color, label=label_name)
                    legend_patches.append(patch)
            
            axes[1].legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3)
            
            # --- Texte en bas ---
            fig.text(0.5, 0.05, score_text, 
                     ha='center', va='center', 
                     fontsize=16, fontweight='bold', color=text_color,
                     bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray'))

            # Ordre important : tight_layout d'abord, PUIS ajustement manuel pour le bas
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15)


            
            # Sauvegarde et Fermeture
            pdf.savefig(fig) 
            plt.close(fig) 
            
            print(f"Page {i+1} ajoutée au PDF.")

        print("Terminé ! Le fichier 'rapport_segmentation.pdf' est prêt.")

def calculer_metrics_segmentation(pred_mask, true_mask, num_classes=18):
    """
    Calcule la précision globale et la mIoU pour une segmentation multi-classes.
    
    Args:
        pred_mask (numpy array): Ta prédiction (2D, valeurs 0-17)
        true_mask (numpy array): La vérité terrain (2D, valeurs 0-17, MÊME TAILLE !)
        num_classes (int): Nombre total de classes possibles (18 pour toi).
        
    Returns:
        pixel_acc (float): Précision globale (attention au piège du fond - le background fait que la segmentation peut sembler assez bonne).
        miou (float): Mean Intersection over Union (le score principal).
        iou_par_classe (array): Le score IoU détaillé pour chaque classe.
    """
    # 1. Aplatir les images en 1D (nécessaire pour la matrice de confusion)
    # On convertit en entiers pour être sûr
    y_pred = pred_mask.flatten().astype(np.int32)
    y_true = true_mask.flatten().astype(np.int32)

    # 2. Créer la Matrice de Confusion
    # C'est une grille 18x18 qui compte les croisements :
    # "Combien de fois le vrai pixel était X mais l'IA a prédit Y ?"
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

    # 3. Calculer les Intersections et Unions à partir de la matrice
    
    # Intersection : C'est la diagonale de la matrice (quand pred == true)
    intersection = np.diag(cm)
    
    # Union : C'est la somme de la ligne + somme de la colonne - l'intersection (pour ne pas la compter deux fois)
    ground_truth_set = cm.sum(axis=1) # Total réel pour chaque classe
    predicted_set = cm.sum(axis=0)    # Total prédit pour chaque classe
    union = ground_truth_set + predicted_set - intersection

    # 4. Calculer l'IoU par classe
    # On utilise un contexte numpy pour éviter les warnings si une classe est absente (division par 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        iou_par_classe = intersection / union
    
    # Si une classe n'existe pas dans l'image, l'union est 0, ce qui donne un NaN (Not a Number).
    # On remplace ces NaNs par 0.0 pour pouvoir faire la moyenne.
    iou_par_classe = np.nan_to_num(iou_par_classe)

    # 5. Calculer les scores finaux
    # mIoU : Moyenne des IoU de toutes les classes
    miou = np.mean(iou_par_classe)
    
    # Précision Globale : Total des bons pixels / Total des pixels
    pixel_acc = np.sum(intersection) / np.sum(cm)
    
    return pixel_acc, miou, iou_par_classe

#variables globales
image_dir = "./images_a_segmenter/top_influenceurs_2024/IMG"  # nom du répertoire contenant les images
image_dir_masks = "./images_a_segmenter/top_influenceurs_2024/Mask" # nom du répertoire contenant les masks
# Charge les variables. Si le .env est dans le même dossier que le notebook, c'est direct.
load_dotenv()
# Récupérer le token Hugging Face
api_token = os.getenv("HUGGING_FACE_TOKEN")

API_URL = "https://router.huggingface.co/models/sayeed99/segformer-b3-clothes" # Remplacez ... par le bon endpoint.
headers = {
    "Authorization": f"Bearer {api_token}"
    # Le "Content-Type" sera ajouté dynamiquement lors de l'envoi de l'image
}

CLASS_MAPPING = {
    "Background": 0,
    "Hat": 1,
    "Hair": 2,
    "Sunglasses": 3,
    "Upper-clothes": 4,
    "Skirt": 5,
    "Pants": 6,
    "Dress": 7,
    "Belt": 8,
    "Left-shoe": 9,
    "Right-shoe": 10,
    "Face": 11,
    "Left-leg": 12,
    "Right-leg": 13,
    "Left-arm": 14,
    "Right-arm": 15,
    "Bag": 16,
    "Scarf": 17
}


def main():

    if api_token:
        print("✅ Le token Hugging Face a été chargé depuis le .env.")
        # Vous pouvez maintenant utiliser hf_token pour vous connecter à Hugging Face
        # depuis ce Notebook.
    else:
        print("❌ Erreur : Le token n'a pas été trouvé. Vérifiez le fichier .env.")


    # Contient tout les éléments du répertoire, ce n'est pas filtré
    image_paths = [] 

    try:
        image_paths = os.listdir(image_dir)
    except FileNotFoundError:
        print(f"Erreur : Le répertoire '{image_dir}' n'a pas été trouvé.")
        image_paths = []

    # Filtrer pour ne garder que les fichiers dont le nom se termine par '.png'
    noms_fichiers_png = [
        element for element in image_paths
        if element.lower().endswith('.png') # .lower() pour une recherche insensible à la casse
        and os.path.isfile(os.path.join(image_dir, element)) # S'assurer que c'est bien un fichier
    ]

    if not image_paths:
        print(f"Aucune image trouvée dans '{image_dir}'. Veuillez y ajouter des images.")
    else:
        print(f"{len(noms_fichiers_png)} image(s) à traiter : {noms_fichiers_png}")

    # Appeler la fonction pour segmenter les images listées dans image_paths
    if noms_fichiers_png:
        print(f"\nTraitement de {len(noms_fichiers_png)} image(s) en batch...")
        batch_seg_results = segment_images_batch(noms_fichiers_png)
        print("Traitement en batch terminé.")
    else:
        batch_seg_results = []
        print("Aucune image à traiter en batch.")

    # Afficher les résultats du batch
    if batch_seg_results:
        display_segmented_images_batch(noms_fichiers_png, batch_seg_results)
    else:
        print("Aucun résultat de segmentation à afficher.")



if __name__ == "__main__":
    main()