/* Edge Impulse Arduino examples
 * Copyright (c) 2022 EdgeImpulse Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * 
 * Original Code by Edge Impulse
 * - 29May23 - This code was adapted by Marcelo Rovai to run on a XIAO ESP32S3 
 * - 06Aug25 - Enhanced with OLED display post-processing for KWS results
 * - Tested with ESP32 by Espressif Systems Core 2.0.17
 * 
 * 
 */

// If your target is limited in memory remove this macro to save 10K RAM
#define EIDSP_QUANTIZE_FILTERBANK   0

/*
 ** NOTE: If you run into TFLite arena allocation issue.
 **
 ** This may be due to may dynamic memory fragmentation.
 ** Try defining "-DEI_CLASSIFIER_ALLOCATION_STATIC" in boards.local.txt (create
 ** if it doesn't exist) and copy this file to
 ** `<ARDUINO_CORE_INSTALL_PATH>/arduino/hardware/<mbed_core>/<core_version>/`.
 **
 ** See
 ** (https://support.arduino.cc/hc/en-us/articles/360012076960-Where-are-the-installed-cores-located-)
 ** to find where Arduino installs cores on your machine.
 **
 ** If the problem persists then there's not enough memory for this model and application.
 */

/* Includes ---------------------------------------------------------------- */
#include <XIAO-ESP32S3-KWS_inferencing.h>
#include <I2S.h>
#include <U8g2lib.h>
#include <Wire.h>

// Audio configuration
#define SAMPLE_RATE 16000U
#define SAMPLE_BITS 16

// OLED Display configuration
U8G2_SSD1306_72X40_ER_1_HW_I2C u8g2(U8G2_R2, U8X8_PIN_NONE);

// Display configuration
#define CONFIDENCE_THRESHOLD 0.6  // Minimum confidence to display result
#define DISPLAY_DURATION 2000     // How long to show result (milliseconds)
#define LED_PIN 21               // Built-in LED pin

/** Audio buffers, pointers and selectors */
typedef struct {
    int16_t *buffer;
    uint8_t buf_ready;
    uint32_t buf_count;
    uint32_t n_samples;
} inference_t;

static inference_t inference;
static const uint32_t sample_buffer_size = 2048;
static signed short sampleBuffer[sample_buffer_size];
static bool debug_nn = false; // Set this to true to see e.g. features generated from the raw signal
static bool record_status = true;

// Display variables
unsigned long last_detection_time = 0;
String last_detected_word = "";
float last_confidence = 0.0;
bool display_result = false;

/**
 * @brief      Initialize OLED display
 */
void initDisplay() {
    u8g2.begin();
    u8g2.clearDisplay();
    
    // Show startup screen
    u8g2.firstPage();
    do {
        u8g2.setFont(u8g2_font_ncenB08_tr);
        u8g2.setCursor(8, 15);
        u8g2.print("KWS");
        u8g2.setCursor(5, 30);
        u8g2.print("Ready");
        u8g2.drawFrame(1, 1, 70, 38);
    } while (u8g2.nextPage());
    
    delay(2000);
    u8g2.clearDisplay();
}

/**
 * @brief      Update OLED display with inference results
 */
void updateDisplay(ei_impulse_result_t* result) {
    // Find the class with highest confidence
    float max_confidence = 0;
    String detected_class = "none";
    int max_index = -1;
    
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        if (result->classification[ix].value > max_confidence) {
            max_confidence = result->classification[ix].value;
            detected_class = String(result->classification[ix].label);
            max_index = ix;
        }
    }
    
    // Only update display if confidence is above threshold
    if (max_confidence > CONFIDENCE_THRESHOLD) {
        last_detected_word = detected_class;
        last_confidence = max_confidence;
        last_detection_time = millis();
        display_result = true;
        
        // Turn on LED for detected keyword (except for "noise" or "unknown")
        if (detected_class != "noise" && detected_class != "unknown") {
            digitalWrite(LED_PIN, LOW);  // LED ON (inverted logic)
        }
    }
    
    // Clear display first
    u8g2.firstPage();
    do {
        // Draw border
        u8g2.drawFrame(0, 0, 72, 40);
        
        if (display_result && (millis() - last_detection_time < DISPLAY_DURATION)) {
            // Display detected word
            u8g2.setFont(u8g2_font_ncenB08_tr);
            
            // Center the text based on word length
            int text_width = u8g2.getStrWidth(last_detected_word.c_str());
            int x_pos = (72 - text_width) / 2;
            u8g2.setCursor(x_pos, 15);
            u8g2.print(last_detected_word);
            
            // Display confidence percentage
            String conf_text = String((int)(last_confidence * 100)) + "%";
            int conf_width = u8g2.getStrWidth(conf_text.c_str());
            int conf_x = (72 - conf_width) / 2;
            u8g2.setCursor(conf_x, 30);
            u8g2.print(conf_text);
            
            // Add small indicator dots based on confidence level
            int dots = (int)(last_confidence * 5); // 0-5 dots
            for (int i = 0; i < dots; i++) {
                u8g2.drawPixel(26 + i * 4, 35);
            }
            
        } else {
            // Show listening state
            u8g2.setFont(u8g2_font_6x10_tr);
            u8g2.setCursor(10, 15);
            u8g2.print("Listening");
            
            // Add animated dots to show it's active
            int dot_count = (millis() / 500) % 4; // 0-3 dots, changes every 500ms
            for (int i = 0; i < dot_count; i++) {
                u8g2.setCursor(12 + i * 6, 28);
                u8g2.print(".");
            }
            
            // Turn off LED when not detecting keywords
            digitalWrite(LED_PIN, HIGH); // LED OFF (inverted logic)
            display_result = false;
        }
        
        // Show timing info (small text at bottom)
        u8g2.setFont(u8g2_font_4x6_tr);
        String timing = String(result->timing.classification) + "ms";
        u8g2.setCursor(2, 38);
        u8g2.print(timing);
        
    } while (u8g2.nextPage());
}

/**
 * @brief      Arduino setup function
 */
void setup()
{
    // put your setup code here, to run once:
    Serial.begin(115200);
    // comment out the below line to cancel the wait for USB connection (needed for native USB)
    while (!Serial);
    Serial.println("Edge Impulse Inferencing Demo with OLED Display");

    // Initialize LED
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, HIGH); // LED OFF initially
    
    // Initialize OLED display
    initDisplay();
    
    // Initialize I2S for microphone
    I2S.setAllPins(-1, 42, 41, -1, -1);
    if (!I2S.begin(PDM_MONO_MODE, SAMPLE_RATE, SAMPLE_BITS)) {
        Serial.println("Failed to initialize I2S!");
        
        // Show error on display
        u8g2.firstPage();
        do {
            u8g2.setFont(u8g2_font_ncenB08_tr);
            u8g2.setCursor(8, 15);
            u8g2.print("I2S");
            u8g2.setCursor(5, 30);
            u8g2.print("Error");
            u8g2.drawFrame(1, 1, 70, 38);
        } while (u8g2.nextPage());
        
        while (1);
    }
    
    // summary of inferencing settings (from model_metadata.h)
    ei_printf("Inferencing settings:\n");
    ei_printf("\tInterval: ");
    ei_printf_float((float)EI_CLASSIFIER_INTERVAL_MS);
    ei_printf(" ms.\n");
    ei_printf("\tFrame size: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    ei_printf("\tSample length: %d ms.\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT / 16);
    ei_printf("\tNo. of classes: %d\n", sizeof(ei_classifier_inferencing_categories) / sizeof(ei_classifier_inferencing_categories[0]));

    ei_printf("\nStarting continuous inference in 2 seconds...\n");
    ei_sleep(2000);

    if (microphone_inference_start(EI_CLASSIFIER_RAW_SAMPLE_COUNT) == false) {
        ei_printf("ERR: Could not allocate audio buffer (size %d), this could be due to the window length of your model\r\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT);
        
        // Show error on display
        u8g2.firstPage();
        do {
            u8g2.setFont(u8g2_font_6x10_tr);
            u8g2.setCursor(5, 15);
            u8g2.print("Memory");
            u8g2.setCursor(8, 30);
            u8g2.print("Error");
            u8g2.drawFrame(1, 1, 70, 38);
        } while (u8g2.nextPage());
        
        return;
    }

    ei_printf("Recording...\n");
}

/**
 * @brief      Arduino main function. Runs the inferencing loop.
 */
void loop()
{
    bool m = microphone_inference_record();
    if (!m) {
        ei_printf("ERR: Failed to record audio...\n");
        return;
    }

    signal_t signal;
    signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
    signal.get_data = &microphone_audio_signal_get_data;
    ei_impulse_result_t result = { 0 };

    EI_IMPULSE_ERROR r = run_classifier(&signal, &result, debug_nn);
    if (r != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", r);
        return;
    }

    // Update OLED display with results
    updateDisplay(&result);

    // print the predictions to serial (optional, for debugging)
    if (debug_nn) {
        ei_printf("Predictions ");
        ei_printf("(DSP: %d ms., Classification: %d ms., Anomaly: %d ms.)",
            result.timing.dsp, result.timing.classification, result.timing.anomaly);
        ei_printf(": \n");
        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            ei_printf("    %s: ", result.classification[ix].label);
            ei_printf_float(result.classification[ix].value);
            ei_printf("\n");
        }
    #if EI_CLASSIFIER_HAS_ANOMALY == 1
        ei_printf("    anomaly score: ");
        ei_printf_float(result.anomaly);
        ei_printf("\n");
    #endif
    }
}

static void audio_inference_callback(uint32_t n_bytes)
{
    for(int i = 0; i < n_bytes>>1; i++) {
        inference.buffer[inference.buf_count++] = sampleBuffer[i];

        if(inference.buf_count >= inference.n_samples) {
          inference.buf_count = 0;
          inference.buf_ready = 1;
        }
    }
}

static void capture_samples(void* arg) {

  const int32_t i2s_bytes_to_read = (uint32_t)arg;
  size_t bytes_read = i2s_bytes_to_read;

  while (record_status) {

    /* read data at once from i2s */
    esp_i2s::i2s_read(esp_i2s::I2S_NUM_0, (void*)sampleBuffer, i2s_bytes_to_read, &bytes_read, 100);

    if (bytes_read <= 0) {
      ei_printf("Error in I2S read : %d", bytes_read);
    }
    else {
        if (bytes_read < i2s_bytes_to_read) {
        ei_printf("Partial I2S read");
        }

        // scale the data (otherwise the sound is too quiet)
        for (int x = 0; x < i2s_bytes_to_read/2; x++) {
            sampleBuffer[x] = (int16_t)(sampleBuffer[x]) * 8;
        }

        if (record_status) {
            audio_inference_callback(i2s_bytes_to_read);
        }
        else {
            break;
        }
    }
  }
  vTaskDelete(NULL);
}

/**
 * @brief      Init inferencing struct and setup/start PDM
 *
 * @param[in]  n_samples  The n samples
 *
 * @return     { description_of_the_return_value }
 */
static bool microphone_inference_start(uint32_t n_samples)
{
    inference.buffer = (int16_t *)malloc(n_samples * sizeof(int16_t));

    if(inference.buffer == NULL) {
        return false;
    }

    inference.buf_count  = 0;
    inference.n_samples  = n_samples;
    inference.buf_ready  = 0;

    ei_sleep(100);

    record_status = true;

    xTaskCreate(capture_samples, "CaptureSamples", 1024 * 32, (void*)sample_buffer_size, 10, NULL);

    return true;
}

/**
 * @brief      Wait on new data
 *
 * @return     True when finished
 */
static bool microphone_inference_record(void)
{
    bool ret = true;

    while (inference.buf_ready == 0) {
        delay(10);
    }

    inference.buf_ready = 0;
    return ret;
}

/**
 * Get raw audio signal data
 */
static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr)
{
    numpy::int16_to_float(&inference.buffer[offset], out_ptr, length);

    return 0;
}

/**
 * @brief      Stop PDM and release buffers
 */
static void microphone_inference_end(void)
{
    free(sampleBuffer);
    ei_free(inference.buffer);
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE
#error "Invalid model for current sensor."
#endif