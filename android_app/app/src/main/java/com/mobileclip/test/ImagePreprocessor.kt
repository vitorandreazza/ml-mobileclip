package com.mobileclip.test

import android.graphics.Bitmap

/**
 * Preprocesses images for MobileCLIP-S2 model inference.
 *
 * Pipeline: Resize -> Center Crop -> ToTensor (normalize to [0,1])
 * - Target size: 256x256
 * - NO mean/std normalization (MobileCLIP doesn't use it)
 */
class ImagePreprocessor {
    companion object {
        private const val IMAGE_SIZE = 256

        /**
         * Preprocesses a bitmap for model inference.
         * Returns a FloatArray in CHW format (channels × height × width).
         */
        fun preprocess(bitmap: Bitmap): FloatArray {
            // Step 1: Resize to 256x256
            val resized = Bitmap.createScaledBitmap(
                bitmap,
                IMAGE_SIZE,
                IMAGE_SIZE,
                true // Use bilinear filtering
            )

            // Step 2: Center crop (already 256x256, so this is a no-op)
            val cropped = centerCrop(resized, IMAGE_SIZE)

            // Step 3: Convert to float array (0-1 range)
            return bitmapToFloatArray(cropped)
        }

        /**
         * Center crops the image to the target size.
         */
        private fun centerCrop(bitmap: Bitmap, size: Int): Bitmap {
            val width = bitmap.width
            val height = bitmap.height

            if (width == size && height == size) {
                return bitmap
            }

            val x = (width - size) / 2
            val y = (height - size) / 2

            return Bitmap.createBitmap(bitmap, x, y, size, size)
        }

        /**
         * Converts bitmap to float array in CHW format with values in [0, 1] range.
         * Shape: [3, 256, 256]
         * NO normalization with mean/std is applied (MobileCLIP doesn't use it).
         */
        private fun bitmapToFloatArray(bitmap: Bitmap): FloatArray {
            val width = bitmap.width
            val height = bitmap.height
            val pixels = IntArray(width * height)

            bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

            // Output array: 3 channels × height × width
            val output = FloatArray(3 * height * width)

            for (i in pixels.indices) {
                val pixel = pixels[i]

                // Extract RGB values and convert to [0, 1] range
                val r = ((pixel shr 16) and 0xFF) / 255.0f
                val g = ((pixel shr 8) and 0xFF) / 255.0f
                val b = (pixel and 0xFF) / 255.0f

                // Store in CHW format (all R, then all G, then all B)
                output[i] = r                          // Red channel
                output[height * width + i] = g         // Green channel
                output[2 * height * width + i] = b     // Blue channel
            }

            return output
        }
    }
}
