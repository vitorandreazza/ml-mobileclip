import open_clip
import torch
from PIL import Image
from executorch.runtime import Runtime

from mobileclip import reparameterize_model


def test_executorch_inference(image_path, texts):
    model_name = "MobileCLIP-S2"
    original_model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='datacompdr')
    original_model.eval()
    original_model = reparameterize_model(original_model)

    runtime = Runtime.get()

    image_pte = f"./models/{model_name.lower()}_visual.pte"
    text_pte = f"./models/{model_name.lower()}_text.pte"

    image_program = runtime.load_program(image_pte)
    text_program = runtime.load_program(text_pte)

    image_method = image_program.load_method("forward")
    text_method = text_program.load_method("forward")

    image = preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0)

    tokenizer = open_clip.get_tokenizer(model_name)
    text_tokens = tokenizer(texts)  # Shape: (len(texts), 77)

    with torch.no_grad():
        image_features_et = image_method.execute([image])[0]

        text_features_et = []
        for i in range(text_tokens.shape[0]):
            single_token = text_tokens[i:i + 1]  # (1, 77)
            feat = text_method.execute([single_token])[0]
            text_features_et.append(feat)
        text_features_et = torch.cat(text_features_et, dim=0)

        image_features_et /= image_features_et.norm(dim=-1, keepdim=True)
        text_features_et /= text_features_et.norm(dim=-1, keepdim=True)

        similarities = 100.0 * (image_features_et @ text_features_et.T)
        probs_et = similarities.softmax(dim=-1).squeeze(0).tolist()

    # Verify against original PyTorch model
    with torch.no_grad():
        image_features_orig = original_model.encode_image(image)
        text_features_orig = original_model.encode_text(text_tokens)

        image_features_orig /= image_features_orig.norm(dim=-1, keepdim=True)
        text_features_orig /= text_features_orig.norm(dim=-1, keepdim=True)

        probs_orig = (100.0 * (image_features_orig @ text_features_orig.T)).softmax(dim=-1).squeeze(0).tolist()

    print("ExecuTorch Probabilities:")
    for text, prob in zip(texts, probs_et):
        print(f"{text}: {prob:.4f}")

    print("\nOriginal PyTorch Probabilities (for verification):")
    for text, prob in zip(texts, probs_orig):
        print(f"{text}: {prob:.4f}")

    if all(abs(p_et - p_orig) < 1e-4 for p_et, p_orig in zip(probs_et, probs_orig)):
        print("\nTest Passed: Outputs match within tolerance.")
    else:
        print("\nWarning: Outputs differ; check export process.")

    # Top match from ExecuTorch
    top_idx = probs_et.index(max(probs_et))
    print(f"\nTop Match (ExecuTorch): {texts[top_idx]} (Probability: {probs_et[top_idx]:.4f})")


if __name__ == "__main__":
    texts = ["a bird", "a cat", "a black cat", "a white cat", "a dog", "a bicycle"]

    test_executorch_inference("test_images/cat.jpeg", texts)
    test_executorch_inference("test_images/dog.jpg", texts)
    test_executorch_inference("test_images/bicycle.jpg", texts)
