# MobileCLIP Android Test App

This Android application tests the MobileCLIP model exported to ExecuTorch format. It performs the same inference tests as the Python `test_exported_model.py` script, but running natively on Android using Kotlin.

## Overview

The app demonstrates:
- Loading and running ExecuTorch `.pte` models on Android
- Image preprocessing (resize, center crop, normalize)
- Text tokenization using CLIP's BPE tokenizer
- Computing image-text similarities
- Displaying classification results

## Requirements

- Android Studio Arctic Fox or newer
- Android SDK 26 (Android 8.0) or higher
- Kotlin 1.9.20+
- MobileCLIP-S2 model files (`.pte` format)

## Project Structure

```
android_app/
├── app/
│   ├── src/main/
│   │   ├── java/com/mobileclip/test/
│   │   │   ├── MainActivity.kt           # Main activity with UI
│   │   │   ├── MobileCLIPInference.kt    # Model loading and inference
│   │   │   ├── ImagePreprocessor.kt      # Image preprocessing pipeline
│   │   │   └── CLIPTokenizer.kt          # BPE text tokenization
│   │   ├── res/
│   │   │   ├── layout/
│   │   │   │   └── activity_main.xml     # UI layout
│   │   │   └── values/
│   │   │       └── strings.xml
│   │   └── assets/
│   │       ├── models/                    # Place .pte files here
│   │       │   ├── mobileclip-s2_visual.pte
│   │       │   └── mobileclip-s2_text.pte
│   │       ├── images/                    # Test images
│   │       │   └── cat.jpeg
│   │       ├── clip-vocab.json            # CLIP vocabulary
│   │       └── clip-merges.txt            # BPE merges
│   └── build.gradle
├── build.gradle
├── settings.gradle
└── README.md
```

## Setup Instructions

### 1. Export MobileCLIP Models

First, you need to export the MobileCLIP models to ExecuTorch format using the Python script:

```bash
cd /home/user/ml-mobileclip/export
python mobileclip_export.py
```

This will generate two `.pte` files in `export/models/`:
- `mobileclip-s2_visual.pte` (image encoder)
- `mobileclip-s2_text.pte` (text encoder)

### 2. Copy Model Files

Copy the exported `.pte` files to the Android app's assets folder:

```bash
mkdir -p android_app/app/src/main/assets/models
cp export/models/mobileclip-s2_visual.pte android_app/app/src/main/assets/models/
cp export/models/mobileclip-s2_text.pte android_app/app/src/main/assets/models/
```

### 3. Add Test Images

The project already includes `cat.jpeg` in the assets. To add more test images:

```bash
# Copy additional test images
cp export/test_images/*.jpeg android_app/app/src/main/assets/images/
```

Then update `MainActivity.kt` to include the new images:

```kotlin
val imageAssetPaths = listOf(
    "images/cat.jpeg",
    "images/dog.jpeg",  // Add your images here
    "images/bird.jpeg"
)
```

### 4. Open in Android Studio

1. Open Android Studio
2. Select "Open an Existing Project"
3. Navigate to `/home/user/ml-mobileclip/android_app`
4. Wait for Gradle to sync dependencies

### 5. Build and Run

1. Connect an Android device or start an emulator
2. Click "Run" (green play button) or press Shift+F10
3. Select your device
4. The app will install and launch

## Usage

1. Launch the app
2. Tap the "Run MobileCLIP Test" button
3. Wait for the models to load and inference to complete
4. View the results displayed on screen

The app will show:
- Loading status
- Probabilities for each text description
- The top match for each image

## Implementation Details

### Image Preprocessing

The `ImagePreprocessor` class implements the same preprocessing pipeline as OpenCLIP:

1. **Resize**: Scale to 256x256 using bilinear interpolation
2. **Center Crop**: Crop center 256x256 region
3. **Normalize**: Apply per-channel normalization
   - Mean: `[0.48145466, 0.4578275, 0.40821073]`
   - Std: `[0.26862954, 0.26130258, 0.27577711]`
4. **Format**: Output as CHW (channels × height × width) float array

### Text Tokenization

The `CLIPTokenizer` class implements CLIP's BPE tokenizer:

1. **Lowercase**: Convert text to lowercase
2. **Regex Split**: Split into tokens using CLIP's pattern
3. **Byte Encode**: Convert to GPT-2 byte encoding
4. **BPE**: Apply Byte Pair Encoding merges
5. **Add Special Tokens**:
   - SOT (Start of Text): token ID 49406
   - EOT (End of Text): token ID 49407
6. **Pad**: Pad to 77 tokens with token ID 0

### Model Inference

The `MobileCLIPInference` class handles:

1. **Model Loading**: Load `.pte` files using ExecuTorch
2. **Image Encoding**: Process image through visual encoder (output: 512-dim)
3. **Text Encoding**: Process text through text encoder (output: 512-dim)
4. **Feature Normalization**: L2 normalize embeddings
5. **Similarity Computation**: Dot product of normalized features
6. **Softmax**: Convert similarities to probabilities

## Testing

The test should produce results matching the Python version:

**Expected Output (for cat.jpeg):**
```
=== cat.jpeg ===

Probabilities:
  a bird: 0.0001
  a cat: 0.9997
  a dog: 0.0002

Top Match: a cat (0.9997)
```

## Troubleshooting

### Models not loading

- Verify `.pte` files are in `app/src/main/assets/models/`
- Check file names match: `mobileclip-s2_visual.pte` and `mobileclip-s2_text.pte`
- Ensure models were exported correctly

### Gradle sync issues

- Update Android Studio to the latest version
- Check internet connection (for downloading dependencies)
- Try "File > Invalidate Caches / Restart"

### Runtime errors

- Check minimum SDK version (26+)
- Verify ExecuTorch dependency is properly resolved
- Check device/emulator compatibility

## Performance Notes

- Model loading takes 1-3 seconds on first run
- Image inference: ~50-200ms per image
- Text inference: ~20-100ms per text
- Performance varies by device

## Dependencies

- **ExecuTorch**: 0.3.0 - PyTorch mobile runtime
- **Kotlin Coroutines**: 1.7.3 - Async operations
- **AndroidX**: Material Design and ConstraintLayout

## Comparison with Python Test

This Android app replicates the functionality of `export/test_exported_model.py`:

| Feature | Python | Android |
|---------|--------|---------|
| Image preprocessing | OpenCLIP transforms | Custom Kotlin implementation |
| Text tokenization | OpenCLIP tokenizer | Custom BPE implementation |
| Model format | ExecuTorch .pte | ExecuTorch .pte |
| Inference | Python ExecuTorch | Android ExecuTorch |
| Results | Command line | On-screen display |

## License

Same as parent MobileCLIP project.

## References

- [MobileCLIP](https://github.com/apple/ml-mobileclip)
- [ExecuTorch](https://pytorch.org/executorch/)
- [CLIP Tokenizer](https://github.com/openai/CLIP)
