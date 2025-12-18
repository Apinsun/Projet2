import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from tqdm import tqdm
import io
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import time
from sklearn.metrics import confusion_matrix
from huggingface_hub.utils import HfHubHTTPError

def png_from_dir(image_path) :
    """
    renvoie la liste des fichiers png dans le répertoire donné en argument
    
    Args image_path(str): le chemin du répertoire à explorer

    Returns (str[]): retourne une liste contenant le noms des fichiers se terminant par .png
    """
        # Contient tout les éléments du répertoire, ce n'est pas filtré
    image_paths = [] 

    try:
        image_paths = os.listdir(image_path)
    except FileNotFoundError:
        print(f"Erreur : Le répertoire '{image_path}' n'a pas été trouvé.")
        image_paths = []

    # Filtrer pour ne garder que les fichiers dont le nom se termine par '.png'
    noms_fichiers_png = [
        element for element in image_paths
        if element.lower().endswith('.png') # .lower() pour une recherche insensible à la casse
        and os.path.isfile(os.path.join(image_path, element)) # S'assurer que c'est bien un fichier
    ]
    #on trie les fichiers pour qu'ils apparaissent dans l'ordre
    noms_fichiers_png.sort()

    return noms_fichiers_png

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
        # C'est decode_mask_bytes qui fait le redimensionnement à la taille originale !
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
# Configuration des tentatives
    MAX_RETRIES = 5  # Nombre max de tentatives pour une même image
    BASE_WAIT_TIME = 2 # Temps d'attente de base en secondes

    client = InferenceClient(
        provider="hf-inference",
        api_key=api_token,
    )

    for ip in tqdm(list_of_image_paths, desc="Segmentation en cours"):
        image_path = os.path.join(image_dir, ip)
        path_to_send = image_path # Par défaut, on envoie l'originale
        is_temp_file = False # Marqueur pour savoir si on doit supprimer le fichier après
        segmentation_mask = None # Par défaut à None en cas d'échec total
        MAX_DIMENSION = 1024  # Dimension maximale en hauteur comme en largeur de nos images
        TEMP_FILENAME = "temp_resized_buffer.png" # Nom du fichier temporaire

        # --- 1. Préparation de l'image ---
        try:
            original_image = Image.open(image_path)
            image_w, image_h = original_image.size

            # Si l'image est trop grande, on crée le fichier temporaire
            if max(image_w, image_h) > MAX_DIMENSION:
                scale_factor = MAX_DIMENSION / max(image_w, image_h)
                new_size = (int(image_w * scale_factor), int(image_h * scale_factor))
                
                # On redimensionne
                resized_img = original_image.resize(new_size, Image.Resampling.LANCZOS)
                
                # On sauvegarde sur le disque
                resized_img.save(TEMP_FILENAME, format="PNG")
                
                # On change le chemin à envoyer
                path_to_send = TEMP_FILENAME
                is_temp_file = True
        except Exception as e:
            print(f"❌ [Fatal] Impossible de lire {ip}: {e}")
            batch_segmentations.append(None)
            continue

        # --- 2. Boucle de tentative d'appel API ---
        try:
            for attempt in range(MAX_RETRIES):
                try:
                    # Appel API
                    response = client.image_segmentation(
                        path_to_send, 
                        model="sayeed99/segformer_b3_clothes"
                    )
                    
                    # Si succès, on traite et on sort de la boucle "attempt"
                    # /!\ on passe les dimensions originales, create_mask va redimensionner aux proportions originales
                    segmentation_mask = create_masks(response, image_w, image_h)
                    break 

                except HfHubHTTPError as e:
                    # --- Gestion fine des codes HTTP ---
                    
                    # CAS A : Le modèle est en train de charger (Cold Boot)
                    if e.response.status_code == 503:
                        # L'API nous donne souvent une estimation du temps d'attente
                        estimated_time = e.response.headers.get("x-compute-time-left")
                        wait_time = float(estimated_time) if estimated_time else BASE_WAIT_TIME * (attempt + 1)
                        
                        print(f"⏳ {ip}: Modèle en chargement... Attente de {wait_time:.1f}s (Tentative {attempt+1}/{MAX_RETRIES})")
                        time.sleep(wait_time)
                        continue # On recommence la boucle

                    # CAS B : Rate Limit (Trop de requêtes)
                    elif e.response.status_code == 429:
                        wait_time = BASE_WAIT_TIME * (2 ** attempt) # Backoff exponentiel (2s, 4s, 8s...)
                        print(f"✋ {ip}: Rate limit atteint. Pause de {wait_time}s.")
                        time.sleep(wait_time)
                        continue

                    # CAS C : Erreur Fatale (Image trop grosse, Mauvaise requête)
                    elif 400 <= e.response.status_code < 500:
                        print(f"❌ {ip}: Erreur client fatale ({e.response.status_code}). Image ignorée.")
                        print(f"   Détail: {e}")
                        break # On arrête les tentatives pour cette image

                    # CAS D : Erreur Serveur (Crash interne chez HF)
                    elif e.response.status_code >= 500:
                        print(f"⚠️ {ip}: Erreur serveur HF ({e.response.status_code})... On réessaie.")
                        time.sleep(BASE_WAIT_TIME)
                        continue
                
                except Exception as e:
                    # Erreurs non liées à l'API (bug de code, parsing, réseau coupé)
                    print(f"❌ {ip}: Erreur inattendue : {e}")
                    break
            
            # Ajout du résultat (mask ou None si toutes les tentatives ont échoué)
            batch_segmentations.append(segmentation_mask)
        
        finally:
            # --- 3. Nettoyage (Le plus important !) ---
            # Quoi qu'il arrive (succès ou erreur), on supprime le fichier temporaire
            if is_temp_file and os.path.exists(TEMP_FILENAME):
                try:
                    os.remove(TEMP_FILENAME)
                except Exception as e:
                    print(f"⚠️ Impossible de supprimer le fichier temporaire : {e}")

        # Petite pause de courtoisie entre chaque image réussie pour éviter le 429
        time.sleep(0.5)

    return batch_segmentations

def display_segmented_images_batch(original_image_paths, segmentation_masks,noms_masks_png):
    """
    Affiche les images originales et leurs masques segmentés.

    Args:
        original_image_paths (list): Liste des chemins des images originales.
        segmentation_masks (list): Liste des masques segmentés (NumPy arrays).
        noms_masks_png (list): Liste des noms de fichiers des masques de vérité terrain.

    """

    """
    Affiche les images originales et leurs masques segmentés côte à côte ainsi que les masks fournis.
    """
   # On prépare le dictionnaire inverse (ID -> Nom) une seule fois au début
    # (Supposant que class_mapping est du type {'Fond': 0, 'Chapeau': 1})
    id_to_label = {v: k for k, v in CLASS_MAPPING.items()}

# --- Listes pour stocker les métriques globales ---
    all_acc = []
    all_miou = []
    all_dice = []
    num_classes = len(CLASS_MAPPING)
    global_conf_matrix = np.zeros((num_classes, num_classes))

    with PdfPages('rapport_segmentation.pdf') as pdf:
        avancee = 1 # Compteur de pages
        # Boucle principale
        for nfp, segmentation_mask,nmp in zip(original_image_paths, segmentation_masks,noms_masks_png):
            
            # (Initialisation des variables métriques pour cette itération)
            acc, miou, mean_dice = np.nan, np.nan, np.nan

            # 3 colonnes (Originale | Pred | Vérité)
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
            
            if segmentation_mask is None:
                axes[1].text(0.5, 0.5, "Segmentation indisponible", ha='center')
                im = None
            else:
            # On stocke l'objet 'im' pour récupérer les couleurs pour la légende
                im = axes[1].imshow(segmentation_mask, cmap='tab20', interpolation='nearest', vmin=0, vmax=19)
            axes[1].set_title("Prédiction IA")
            axes[1].axis('off')
            
            # --- 3. Vérité Terrain (Droite)  ---
            mask_path = os.path.join(image_dir_masks, nmp) 
            
            try:
                true_mask_pil = Image.open(mask_path)
                
                # Redimensionnement (Nearest Neighbor) si les tailles diffèrent
                # PIL utilise (largeur, hauteur), alors que numpy shape est (hauteur, largeur)
                if true_mask_pil.size != (segmentation_mask.shape[1], segmentation_mask.shape[0]):
                    true_mask_pil = true_mask_pil.resize((segmentation_mask.shape[1], segmentation_mask.shape[0]), resample=Image.NEAREST)
                
                # Conversion en tableau numpy
                true_mask = np.array(true_mask_pil)
                
                axes[2].imshow(true_mask, cmap='tab20', interpolation='nearest', vmin=0, vmax=19)
                axes[2].set_title(f"Vérité Terrain ({nmp})")
                
            except FileNotFoundError:
                axes[2].text(0.5, 0.5, "Masque non trouvé", ha='center')
                axes[2].set_title("Vérité Terrain (Manquante)")
                print(f"Masque de vérité terrain non trouvé pour l'image {nfp} : {mask_path}")
            
            axes[2].axis('off')

        # --- CALCUL DES MÉTRIQUES ---
            if segmentation_mask is not None:
                acc, miou, mean_dice, per_image_conf_matrix = calculer_metrics_segmentation(segmentation_mask, true_mask)
                # Mise à jour de la matrice de confusion globale
                global_conf_matrix += per_image_conf_matrix
                # AJOUT : On stocke les valeurs dans les listes globales
                all_acc.append(acc)
                all_miou.append(miou)
                all_dice.append(mean_dice)
            else:
                # Si pas de segmentation, on peut choisir d'ignorer ou mettre 0
                pass
                
            # Préparer le texte à afficher selon si les métriques existent
            if np.isnan(mean_dice):
                dice_text = "mDice: N/A"
            else:
                dice_text = f"Mean Dice (mDice): {mean_dice:.2%}"

            if np.isnan(acc) or np.isnan(miou):
                score_text = dice_text
            else:
                score_text = f"Global Accuracy: {acc:.2%}  |  Mean IoU (mIoU): {miou:.2%}  |  {dice_text}"
            
            # Petit bonus couleur : Vert si bon score, Rouge si mauvais
            text_color = "darkgreen" if miou > 0.6 else "firebrick"

            # --- Légende (au niveau de la figure, à droite) ---
            legend_patches = []
            if segmentation_mask is not None:
                unique_classes = np.unique(segmentation_mask)
            else:
                unique_classes = []

            # ajout des classes manquantes dans notre prediction
            # 2. Classes de la Vérité Terrain
            # On vérifie si true_mask existe (car il est créé dans un try/except)
            if 'true_mask' in locals() and true_mask is not None:
                classes_gt = np.unique(true_mask)
            else:
                classes_gt = np.array([])
            
            # 3. Union des deux (np.union1d trie et dédoublonne automatiquement)
            union_unique_classes = np.union1d(unique_classes, classes_gt)


            if im is not None:
                for val in union_unique_classes:
                    class_id = int(val)
                    try:
                        normed = im.norm(class_id)
                    except Exception:
                        normed = None
                    if normed is not None:
                        color = im.cmap(normed)
                        label_name = id_to_label.get(class_id, f"Class {class_id}")
                        legend_patches.append(mpatches.Patch(color=color, label=label_name))

            # Place la légende sur la droite de la figure pour ne pas empiéter en bas
            if legend_patches:
                fig.legend(handles=legend_patches, loc='center right',
                           bbox_to_anchor=(0.98, 0.5), frameon=False,
                           ncol=1)
                # Garder un espace suffisant à droite pour la légende
                plt.subplots_adjust(left=0.03, right=0.78, top=0.94, bottom=0.08)
            else:
                plt.subplots_adjust(left=0.03, right=0.98, top=0.94, bottom=0.08)

            # --- Texte de score : bas-centre, bien au-dessus du bord ---
            fig.text(0.5, 0.04, score_text,
                     ha='center', va='center',
                     fontsize=14, fontweight='bold', color=text_color,
                     bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray'))


            
            # Sauvegarde et Fermeture
            pdf.savefig(fig) 
            plt.close(fig) 
            
            print(f"Page {avancee} ajoutée au PDF.")
            avancee+=1

        # ============================================================
        # ===         GÉNÉRATION DU RAPPORT FINAL GLOBAL           ===
        # ============================================================
        
        print("Génération du rapport statistique global...")
        
        # On convertit en numpy array pour faciliter les calculs et gérer les NaNs
        arr_acc = np.array(all_acc)
        arr_miou = np.array(all_miou)
        arr_dice = np.array(all_dice)

        # Création d'une nouvelle figure pour le résumé
        fig_summary = plt.figure(figsize=(20, 12))
        fig_summary.suptitle("Rapport Statistique Global de Segmentation", fontsize=24, fontweight='bold')

        # --- 1. Graphique : Boxplots (Boîtes à moustaches) ---
        # Permet de voir la distribution : Médiane, quartiles, et valeurs extrêmes
        ax1 = fig_summary.add_subplot(2, 1, 1) # Haut
        
        data_to_plot = [arr_acc, arr_miou, arr_dice]
        labels = ['Pixel Accuracy', 'Mean IoU (mIoU)', 'Mean Dice (mDice)']
        
        # On filtre les NaN au cas où
        data_to_plot = [d[~np.isnan(d)] for d in data_to_plot]

        bplot = ax1.boxplot(data_to_plot, patch_artist=True, tick_labels=labels, vert=False)
        
        # Couleurs des boîtes
        colors = ['lightblue', 'lightgreen', 'plum']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        
        ax1.set_title("Distribution des scores sur tout le dataset", fontsize=16)
        ax1.set_xlabel("Score (0.0 à 1.0)", fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # Ajout des points individuels (jitter) pour voir la dispersion réelle
        for i, data in enumerate(data_to_plot):
            y = np.random.normal(i + 1, 0.04, size=len(data))
            ax1.plot(data, y, 'r.', alpha=0.5)

        # --- 2. Tableau des Statistiques ---
        ax2 = fig_summary.add_subplot(2, 1, 2) # Bas
        ax2.axis('off') # On cache les axes classiques pour dessiner un tableau
        
        # Calcul des stats
        rows = ['Moyenne', 'Médiane', 'Minimum', 'Maximum', 'Écart-type (Std)']
        cols = ['Pixel Accuracy', 'mIoU', 'mDice']
        
        cell_text = []
        for d in [arr_acc, arr_miou, arr_dice]:
            # On utilise np.nanmean, etc pour ignorer les erreurs
            if len(d) > 0:
                cell_text.append([
                    f"{np.nanmean(d):.2%}",
                    f"{np.nanmedian(d):.2%}",
                    f"{np.nanmin(d):.2%}",
                    f"{np.nanmax(d):.2%}",
                    f"{np.nanstd(d):.4f}"
                ])
            else:
                cell_text.append(["N/A"] * 5)
        
        # Le tableau attend les données transposées (Lignes = Stats, Colonnes = Métriques)
        cell_text = np.array(cell_text).T 

        # Création du tableau
        table = ax2.table(cellText=cell_text,
                          rowLabels=rows,
                          colLabels=cols,
                          cellLoc='center',
                          loc='center',
                          bbox=[0.1, 0.1, 0.8, 0.8]) # Centré
        
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1, 2) # Agrandir les cellules verticalement

        # Coloriage léger des en-têtes
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('darkblue')
            elif j == -1: # Row labels
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#f2f2f2')

        pdf.savefig(fig_summary)
        plt.close(fig_summary)

        # ============================================================
        # === PAGE DE RÉSUMÉ 2 : PERFORMANCE GLOBALE PAR CLASSE    ===
        # ============================================================
        print("Génération Résumé 2 (Global Metrics)...")
        
        # 1. Calcul des scores globaux depuis la matrice cumulée
        # Formule : IoU = TP / (TP + FP + FN)
        intersection = np.diag(global_conf_matrix) # TP
        ground_truth_set = global_conf_matrix.sum(axis=1) # TP + FN
        predicted_set = global_conf_matrix.sum(axis=0)    # TP + FP
        union = ground_truth_set + predicted_set - intersection
        
        # IoU par classe (évite division par 0)
        iou_per_class = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union!=0)
        
        # Global mIoU (Moyenne des IoU valides seulement)
        valid_classes = union > 0
        global_miou_score = np.mean(iou_per_class[valid_classes])

        # 2. Création de la figure
        fig_global = plt.figure(figsize=(20, 12))
        
        # Titre Principal
        fig_global.suptitle(f"Résumé 2 : Performance Globale (Global mIoU: {global_miou_score:.2%})", 
                            fontsize=22, fontweight='bold', color='darkblue')

        ax_table = fig_global.add_subplot(1, 1, 1)
        ax_table.axis('off')

        # 3. Préparation des données pour le tableau
        table_data = []
        # En-têtes
        col_labels = ["ID", "Nom de la Classe", "IoU Global", "Pixels Totaux (Vérité)"]
        
        for class_id in range(num_classes):
            class_name = id_to_label.get(class_id, f"Class {class_id}")
            iou_val = iou_per_class[class_id]
            pixel_count = ground_truth_set[class_id] # Combien de fois cette classe apparait réellement
            
            # On affiche seulement si la classe existe dans le dataset (ou si on veut tout voir)
            # Ici on affiche tout, mais on met "-" si pas présent
            if pixel_count > 0 or predicted_set[class_id] > 0:
                iou_txt = f"{iou_val:.2%}"
                # Petit indicateur visuel (Vert si > 50%, Rouge sinon)
                status = "[OK]" if iou_val > 0.5 else "[!]"
            else:
                iou_txt = "N/A (Absent)"
                status = "-"

            table_data.append([
                str(class_id),
                class_name,
                f"{iou_txt} {status if status != '-' else ''}",
                f"{int(pixel_count):,}"
            ])

        # 4. Dessin du tableau
        table = ax_table.table(cellText=table_data,
                               colLabels=col_labels,
                               cellLoc='center',
                               loc='center',
                               bbox=[0.1, 0.05, 0.8, 0.9]) # Marges
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5) # Plus aéré

        # Style du header
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#40466e') # Bleu foncé
            elif i > 0 and i % 2 == 0:
                cell.set_facecolor('#f2f2f2') # Zébrure légère pour la lisibilité

        pdf.savefig(fig_global)
        plt.close(fig_global)
        
        print("Rapport PDF généré avec succès (incluant la page de résumé final) !")

def calculer_metrics_segmentation(pred_mask, true_mask, num_classes=18):
    """
    Calcule la précision globale et la mIoU et le mDice pour une segmentation multi-classes.
    
    Args:
        pred_mask (numpy array): La prédiction renvoyé par le modèle (2D, valeurs 0-17)
        true_mask (numpy array): La vérité terrain (2D, valeurs 0-17, MÊME TAILLE que pred_mask!)
        num_classes (int): Nombre total de classes possibles
        
    Returns:
        pixel_acc (float): Précision globale (attention au piège du fond - le background fait que la segmentation peut sembler assez bonne).
        miou (float): Mean Intersection over Union (le score principal).
        iou_par_classe (array): Le score IoU détaillé pour chaque classe.
        mean_dice (float): Dice moyen (mDice).
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
    
    # Intersection : C'est la diagonale de la matrice ( les true positifs)
    intersection = np.diag(cm)
    
    # Union : C'est la somme de la ligne + somme de la colonne - l'intersection (pour ne pas la compter deux fois)
    # L'union représente l'aire qui a été prédite à tort OU qui aurait dû être prédite donc TP + FP + FN
    ground_truth_set = cm.sum(axis=1) # TP + FN
    predicted_set = cm.sum(axis=0)    # TP + FP
    union = ground_truth_set + predicted_set - intersection

    classes_presentes = union > 0 # Booléen : Vrai si la classe est présente dans l'image

    # 4. Calcul IoU
    with np.errstate(divide='ignore', invalid='ignore'):
        iou_par_classe = intersection / union

    # gestion d'erreur, cas où on aurait aucune classe présente
    if np.sum(classes_presentes) == 0:
        miou = 0.0
    else:
        # On fait la moyenne uniquement sur les classes pertinentes
        miou = np.mean(iou_par_classe[classes_presentes])

    # 5. Dice (F1) par classe : 2 * TP / (Zone réelle + Zone prédite)
    with np.errstate(divide='ignore', invalid='ignore'):
        dice_par_classe = (2.0 * intersection) / (predicted_set + ground_truth_set)
    dice_par_classe = np.nan_to_num(dice_par_classe)

    if np.sum(classes_presentes) == 0:
        mean_dice = 0.0
    else:
        mean_dice = np.mean(dice_par_classe[classes_presentes])
    
    # Précision Globale : Total des bons pixels / Total des pixels
    pixel_acc = np.sum(intersection) / np.sum(cm)
    
    return pixel_acc, miou, mean_dice, cm

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


    noms_fichiers_png = png_from_dir(image_dir)
    noms_masks_png = png_from_dir(image_dir_masks)

    if not noms_fichiers_png:
        print(f"Aucune image trouvée dans '{image_dir}'. Veuillez y ajouter des images.")
    else:
        print(f"{len(noms_fichiers_png)} image(s) à traiter")

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
        display_segmented_images_batch(noms_fichiers_png, batch_seg_results,noms_masks_png)
    else:
        print("Aucun résultat de segmentation à afficher.")



if __name__ == "__main__":
    main()