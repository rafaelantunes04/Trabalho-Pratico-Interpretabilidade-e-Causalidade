import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.stats import pearsonr
from tqdm import tqdm
from tf_explain.core.integrated_gradients import IntegratedGradients
from tf_explain.core.smoothgrad import SmoothGrad
from tf_explain.core.gradients_inputs import GradientsInputs
import tensorflow as tf
from tf_explain.core.vanilla_gradients import VanillaGradients
from tf_explain.core.grad_cam import GradCAM


# Configuração Global dos Explainers
EXPLAINERS = {
    'integrated_gradients': IntegratedGradients(),
    'grad_input': GradientsInputs(),
    'smoothgrad': SmoothGrad(),
    'gradcam': GradCAM(),
    'saliency': VanillaGradients()
}

def criar_subset_estratificado(X, y, samples_per_class=100, seed=42):
    """
    Cria um subset estratificado com N amostras por classe.
    """
    # Configuração Inicial
    np.random.seed(seed)
    X_subset = []
    y_subset = []
    classes = np.unique(y)

    # Seleção por Classe
    for c in classes:
        idx_class = np.where(y == c)[0]

        if len(idx_class) < samples_per_class:
            raise ValueError(f"Classe {c} tem apenas {len(idx_class)} amostras.")

        chosen_idx = np.random.choice(
            idx_class,
            size=samples_per_class,
            replace=False
        )

        X_subset.append(X[chosen_idx])
        y_subset.append(y[chosen_idx])

    # Concatenação e Embaralhamento
    X_subset = np.concatenate(X_subset, axis=0)
    y_subset = np.concatenate(y_subset, axis=0)

    perm = np.random.permutation(len(y_subset))
    X_subset = X_subset[perm]
    y_subset = y_subset[perm]

    return X_subset, y_subset


def analise_iterativa_pixels(model, image, label, method_name, top_k=10):
    """
    Remove iterativamente os pixels mais importantes e verifica a queda de confiança.
    """
    # Preparação da Imagem (garante formato (Batch, H, W, C))
    raw_image = np.squeeze(image)
    actual_label = int(label)
    
    img_batch = image[np.newaxis, ...] if image.ndim == 3 else image[np.newaxis, ..., np.newaxis]
    
    # Geração da Explicação
    kwargs = {}
    if method_name == 'integrated_gradients':
        kwargs['n_steps'] = 50
    if method_name == 'smoothgrad':
        kwargs['num_samples'] = 70
        kwargs['noise'] = 0.1
            
    explainer = EXPLAINERS.get(method_name)
    explanation = explainer.explain(
        validation_data=(img_batch, None),
        model=model,
        class_index=actual_label,
        **kwargs
    )
    explanation = np.squeeze(explanation)

    # Pós-processamento
    if method_name in ['integrated_gradients', 'smoothgrad']:
        explanation = explanation * raw_image

    # --- CORREÇÃO AQUI ---
    # Passo 1: Calcular a magnitude (importância absoluta)
    explanation = np.abs(explanation) 
    
    # Passo 2: Normalização (agora entre 0 e 1 baseada em magnitude)
    explanation_norm = (explanation - explanation.min()) / (explanation.max() - explanation.min() + 1e-8)
    
    # Passo 3: Ranking
    H, W = explanation_norm.shape
    sorted_indices = np.argsort(explanation_norm.flatten())[::-1] # Maiores magnitudes primeiro
    ranking = [(int(i // W), int(i % W)) for i in sorted_indices]

    # Loop de Perturbação e Visualização
    image_to_modify = raw_image.copy()
    
    for i in range(top_k):
        row, col = ranking[i]
        image_to_modify[row, col] = 0.0 # Remove pixel (torna preto)
        
        # Predict com a imagem modificada
        input_model = image_to_modify[np.newaxis, ..., np.newaxis] if raw_image.ndim == 2 else image_to_modify[np.newaxis, ...]
        preds = model.predict(input_model, verbose=0)
        
        plt.figure(figsize=(4, 4))
        plt.imshow(image_to_modify, cmap='gray')
        plt.scatter(col, row, color='red', s=20, marker='x')
        plt.title(f"Iter: {i+1} | Pixel: ({row},{col})\nConf Real ({actual_label}): {preds[0, actual_label]:.4f}")
        plt.axis('off')
        plt.show()


def display_explanation(
    model,
    image,
    label,
    method_name,
    threshold=0.2,
    alpha=0.7,
    layer_name=None
):
    """
    Exibe a imagem original com o mapa de explicação sobreposto,
    com pós-processamento consistente para MNIST e imagens RGB.
    """

    # -----------------------------
    # Preparação do input
    # -----------------------------
    if image.ndim == 2:
        img_batch = image[np.newaxis, ..., np.newaxis]
    elif image.ndim == 3:
        img_batch = image[np.newaxis, ...]

    kwargs = {}
    if method_name == "integrated_gradients":
        kwargs["n_steps"] = 100
    if method_name == "smoothgrad":
        kwargs["num_samples"] = 100
        kwargs["noise"] = 0.15
    if method_name == "gradcam" and layer_name:
        kwargs["layer_name"] = layer_name

    explainer = EXPLAINERS.get(method_name)
    if explainer is None:
        raise ValueError(f"Método desconhecido: {method_name}")

    # -----------------------------
    # Geração da explicação
    # -----------------------------
    explanation = explainer.explain(
        validation_data=(img_batch, None),
        model=model,
        class_index=int(label),
        **kwargs
    )
    explanation = np.squeeze(explanation)

    # -----------------------------
    # Pós-processamento correto
    # -----------------------------
    # Magnitude
    explanation = np.abs(explanation)

    # Redução de canais se RGB
    if explanation.ndim == 3:
        explanation = explanation.mean(axis=-1)

    # -----------------------------
    # Normalização robusta
    # -----------------------------
    p_low, p_high = np.percentile(explanation, [1, 99])
    explanation_norm = np.clip(
        (explanation - p_low) / (p_high - p_low + 1e-8),
        0.0,
        1.0
    )

    # Transparência contínua (sem NaN)
    alpha_map = explanation_norm.copy()
    alpha_map[alpha_map < threshold] = 0.0

    if method_name == "saliency":
        explanation_norm = cv2.GaussianBlur(explanation_norm, (5, 5), 0)

        # Re-normalizar após o blur
        explanation_norm = (explanation_norm - explanation_norm.min()) / (explanation_norm.max() - explanation_norm.min() + 1e-8)
    
    # -----------------------------
    # Preparação da imagem base
    # -----------------------------
    img_vis = image.copy()

    if img_vis.max() > 1.0:
        img_vis = img_vis / 255.0

    # -----------------------------
    # Visualização
    # -----------------------------
    plt.figure(figsize=(6, 6))

    if img_vis.ndim == 2:
        plt.imshow(img_vis, cmap="gray")
    else:
        plt.imshow(img_vis)

    plt.imshow(
        explanation_norm,
        cmap="cool",
        alpha=alpha_map * alpha
    )

    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(f"Método: {method_name} | Classe: {label}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()



def analisar_sensitivity_n(
    model,
    images,
    labels,
    method_name="grad_input",
    max_n=30,
    num_samples=20,
    batch_size=32,
    baseline_val=0.0
):
    """
    Calcula a métrica Sensitivity-n: correlação entre a soma das atribuições 
    dos pixels removidos e a queda de confiança do modelo.
    """
    # Configuração do Explicador
    kwargs = {}
    explainer = EXPLAINERS.get(method_name)
    if method_name == 'integrated_gradients':
         kwargs['n_steps'] = 50
    if method_name == 'smoothgrad':
        kwargs['num_samples'] = 70
        kwargs['noise'] = 0.1
    
    # Seleção de Amostras
    n_total = len(images)
    indices = np.random.choice(n_total, min(num_samples, n_total), replace=False)

    selected_images = images[indices]
    selected_labels = labels[indices].astype(int)
    actual_samples = len(indices)

    print(f"--- Sensitivity-n ({method_name}) | {actual_samples} amostras ---")

    # Predições Iniciais (Baseline)
    all_preds = model.predict(selected_images, batch_size=batch_size, verbose=0)
    initial_confs = all_preds[np.arange(actual_samples), selected_labels]

    # Inicialização de Buffers
    confidence_curves = np.zeros((actual_samples, max_n))
    global_sum_attributions = np.zeros(actual_samples * max_n)
    global_conf_drops = np.zeros(actual_samples * max_n)

    sample_shape = selected_images[0].shape
    perturbed_batch = np.zeros((batch_size,) + sample_shape, dtype=images.dtype)

    # Loop Principal (por amostra)
    for i in tqdm(range(actual_samples), desc=f"Processando {method_name}"):

        img = selected_images[i]
        label = selected_labels[i]
        init_conf = initial_confs[i]

        img_input = img[np.newaxis, ...] if img.ndim == 3 else img[np.newaxis, ..., np.newaxis]
        label_tensor = tf.convert_to_tensor(label, tf.int32)

        # Geração da Explicação
        attr_map = explainer.explain(
            validation_data=(img_input, None),
            model=model,
            class_index=label_tensor,
            **kwargs
        )
        attr_map = np.squeeze(attr_map)

        # Pós-processamento (Limpeza de Fundo)
        if method_name in ['integrated_gradients', 'smoothgrad']:
             attr_map = attr_map * np.squeeze(img)

        # Normalização e Ranking
        flat_attr = attr_map.reshape(-1)
        amin, amax = flat_attr.min(), flat_attr.max()
        denom = (amax - amin) if amax != amin else 1e-8
        flat_attr_norm = (flat_attr - amin) / denom

        sorted_idx = np.argsort(flat_attr_norm)[::-1]
        w = img.shape[1]
        rows = sorted_idx // w
        cols = sorted_idx % w

        # Perturbação Cumulativa em Batch
        current_perturbed = img.copy()
        steps_done = 0

        while steps_done < max_n:
            cur_bs = min(batch_size, max_n - steps_done)

            for b in range(cur_bs):
                step = steps_done + b
                r, c = rows[step], cols[step]

                if img.ndim == 3:
                    current_perturbed[r, c, :] = baseline_val
                else:
                    current_perturbed[r, c] = baseline_val

                perturbed_batch[b] = current_perturbed

                # Acumula atribuição até o passo atual
                gidx = i * max_n + step
                global_sum_attributions[gidx] = flat_attr_norm[sorted_idx[:step + 1]].sum()

            # Predição do batch perturbado
            preds = model.predict(
                perturbed_batch[:cur_bs],
                batch_size=cur_bs,
                verbose=0
            )[:, label]

            # Armazena resultados
            confidence_curves[i, steps_done:steps_done + cur_bs] = preds
            global_conf_drops[
                i * max_n + steps_done : i * max_n + steps_done + cur_bs
            ] = init_conf - preds

            steps_done += cur_bs

    # Cálculo de Métricas Finais
    correlation, _ = pearsonr(global_sum_attributions, global_conf_drops)
    avg_curve = confidence_curves.mean(axis=0)
    std_curve = confidence_curves.std(axis=0)

    # Visualização (Plot)
    plt.figure(figsize=(10, 6))
    alpha = min(0.5, 10 / actual_samples)

    # Curvas individuais
    plt.plot(range(1, max_n + 1), confidence_curves.T,
             color="gray", alpha=alpha, linewidth=0.5)

    # Curva média
    plt.plot(range(1, max_n + 1), avg_curve,
             color="red", linewidth=3, label="Média")

    # Desvio padrão
    plt.fill_between(
        range(1, max_n + 1),
        avg_curve - std_curve,
        avg_curve + std_curve,
        color="red",
        alpha=0.2,
        label="Desvio padrão"
    )

    plt.title(f"Sensitivity-n ({method_name})\nCorrelação: {correlation:.4f}")
    plt.xlabel("Pixels removidos (Ordem de Importância)")
    plt.ylabel("Confiança na classe real")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "correlation": correlation,
        "avg_curve": avg_curve,
        "confidence_curves": confidence_curves,
        "global_sum_attributions": global_sum_attributions,
        "global_conf_drops": global_conf_drops
    }