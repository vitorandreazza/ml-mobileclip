package com.mobileclip.test

import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {
    private lateinit var btnRunTest: Button
    private lateinit var tvResults: TextView
    private lateinit var inference: MobileCLIPInference

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        btnRunTest = findViewById(R.id.btnRunTest)
        tvResults = findViewById(R.id.tvResults)

        inference = MobileCLIPInference(this)

        btnRunTest.setOnClickListener {
            runTest()
        }
    }

    private fun runTest() {
        btnRunTest.isEnabled = false
        tvResults.text = "Loading models and running test...\n"

        CoroutineScope(Dispatchers.Main).launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    runTestAsync()
                }
                Log.d("MainActivity", result)
                tvResults.text = result
//            } catch (e: Exception) {
//                tvResults.text = "Error running test:\n${e.message}\n\n${e.stackTraceToString()}"
            } finally {
                btnRunTest.isEnabled = true
            }
        }
    }

    private suspend fun runTestAsync(): String {
        val sb = StringBuilder()

        // Load models
        sb.append("Loading MobileCLIP models...\n")
        val loaded = inference.loadModels()
        if (!loaded) {
            return "Failed to load models. Make sure .pte files are in assets/models/"
        }
        sb.append("Models loaded successfully!\n\n")

        // Define test images and texts (matching the Python test)
        val imageAssetPaths = listOf(
            "images/cat.jpeg",
            "images/dog.jpg",
            "images/bicycle.jpg"
        )

        val texts = listOf("a bird", "a cat", "a black cat", "a white cat", "a dog", "a bicycle")

        // Run inference
        sb.append("Running inference...\n\n")
        val results = inference.runTest(imageAssetPaths, texts)

        // Format and return results
        sb.append(results.formatResults())
        sb.append("\n\nTest completed successfully!")

        return sb.toString()
    }

    override fun onDestroy() {
        super.onDestroy()
        inference.release()
    }
}
