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
 */

// Generic version - works with any Edge Impulse image classification model
// Tested with ESP32 by Espressif Systems Core 2.0.17 on XIAO ESP32S3 Sense V1.0 and v1.1
// Adapted from Edge Impulse ESP32 Camera example
// Marcelo Rovai, August, 5th 2025

/* Includes ---------------------------------------------------------------- */
// Replace this with your actual model library name
#include <Box_versus_Wheel_-_XIAO_ESP32S3_inferencing.h>
#include "edge-impulse-sdk/dsp/image/image.hpp"
#include "esp_camera.h"
#include <U8g2lib.h>
#include <Wire.h>

// XIAO ESP32S3 Sense Camera pin definitions
#define CAMERA_MODEL_XIAO_ESP32S3 // Has PSRAM

// Configuration 1: Most common OV2640 configuration
#define CONFIG_1_PWDN_GPIO_NUM    -1
#define CONFIG_1_RESET_GPIO_NUM   -1
#define CONFIG_1_XCLK_GPIO_NUM    10
#define CONFIG_1_SIOD_GPIO_NUM    40
#define CONFIG_1_SIOC_GPIO_NUM    39
#define CONFIG_1_Y9_GPIO_NUM      48
#define CONFIG_1_Y8_GPIO_NUM      11
#define CONFIG_1_Y7_GPIO_NUM      12
#define CONFIG_1_Y6_GPIO_NUM      14
#define CONFIG_1_Y5_GPIO_NUM      16
#define CONFIG_1_Y4_GPIO_NUM      18
#define CONFIG_1_Y3_GPIO_NUM      17
#define CONFIG_1_Y2_GPIO_NUM      15
#define CONFIG_1_VSYNC_GPIO_NUM   38
#define CONFIG_1_HREF_GPIO_NUM    47
#define CONFIG_1_PCLK_GPIO_NUM    13

// Configuration 2: Alternative OV3660 configuration
#define CONFIG_2_PWDN_GPIO_NUM    -1
#define CONFIG_2_RESET_GPIO_NUM   -1
#define CONFIG_2_XCLK_GPIO_NUM    15
#define CONFIG_2_SIOD_GPIO_NUM    4
#define CONFIG_2_SIOC_GPIO_NUM    5
#define CONFIG_2_Y9_GPIO_NUM      16
#define CONFIG_2_Y8_GPIO_NUM      17
#define CONFIG_2_Y7_GPIO_NUM      18
#define CONFIG_2_Y6_GPIO_NUM      12
#define CONFIG_2_Y5_GPIO_NUM      10
#define CONFIG_2_Y4_GPIO_NUM      8
#define CONFIG_2_Y3_GPIO_NUM      9
#define CONFIG_2_Y2_GPIO_NUM      11
#define CONFIG_2_VSYNC_GPIO_NUM   6
#define CONFIG_2_HREF_GPIO_NUM    7
#define CONFIG_2_PCLK_GPIO_NUM    13

// Configuration 3: Another possible configuration
#define CONFIG_3_PWDN_GPIO_NUM    -1
#define CONFIG_3_RESET_GPIO_NUM   -1
#define CONFIG_3_XCLK_GPIO_NUM    10
#define CONFIG_3_SIOD_GPIO_NUM    4
#define CONFIG_3_SIOC_GPIO_NUM    5
#define CONFIG_3_Y9_GPIO_NUM      48
#define CONFIG_3_Y8_GPIO_NUM      11
#define CONFIG_3_Y7_GPIO_NUM      12
#define CONFIG_3_Y6_GPIO_NUM      14
#define CONFIG_3_Y5_GPIO_NUM      16
#define CONFIG_3_Y4_GPIO_NUM      18
#define CONFIG_3_Y3_GPIO_NUM      17
#define CONFIG_3_Y2_GPIO_NUM      15
#define CONFIG_3_VSYNC_GPIO_NUM   38
#define CONFIG_3_HREF_GPIO_NUM    47
#define CONFIG_3_PCLK_GPIO_NUM    13

// Start with Configuration 1 (most common)
#if defined(CAMERA_MODEL_XIAO_ESP32S3)
#define PWDN_GPIO_NUM    CONFIG_1_PWDN_GPIO_NUM
#define RESET_GPIO_NUM   CONFIG_1_RESET_GPIO_NUM
#define XCLK_GPIO_NUM    CONFIG_1_XCLK_GPIO_NUM
#define SIOD_GPIO_NUM    CONFIG_1_SIOD_GPIO_NUM
#define SIOC_GPIO_NUM    CONFIG_1_SIOC_GPIO_NUM
#define Y9_GPIO_NUM      CONFIG_1_Y9_GPIO_NUM
#define Y8_GPIO_NUM      CONFIG_1_Y8_GPIO_NUM
#define Y7_GPIO_NUM      CONFIG_1_Y7_GPIO_NUM
#define Y6_GPIO_NUM      CONFIG_1_Y6_GPIO_NUM
#define Y5_GPIO_NUM      CONFIG_1_Y5_GPIO_NUM
#define Y4_GPIO_NUM      CONFIG_1_Y4_GPIO_NUM
#define Y3_GPIO_NUM      CONFIG_1_Y3_GPIO_NUM
#define Y2_GPIO_NUM      CONFIG_1_Y2_GPIO_NUM
#define VSYNC_GPIO_NUM   CONFIG_1_VSYNC_GPIO_NUM
#define HREF_GPIO_NUM    CONFIG_1_HREF_GPIO_NUM
#define PCLK_GPIO_NUM    CONFIG_1_PCLK_GPIO_NUM
#else
#error "Camera model not selected"
#endif

/* Constant defines -------------------------------------------------------- */
#define EI_CAMERA_RAW_FRAME_BUFFER_COLS           320
#define EI_CAMERA_RAW_FRAME_BUFFER_ROWS           240
#define EI_CAMERA_FRAME_BYTE_SIZE                 3

// Memory management settings
#define EI_CLASSIFIER_ALLOCATION_HEAP             1
#define EI_CLASSIFIER_ALLOCATION_STATIC_HIMAX     0

// OLED Display initialization
U8G2_SSD1306_72X40_ER_1_HW_I2C u8g2(U8G2_R2, U8X8_PIN_NONE);

/* Private variables ------------------------------------------------------- */
static bool is_initialised = false;
uint8_t *snapshot_buf;

// Variables for classification results
String detected_class = "";
String abbreviated_class = "";
float confidence = 0.0;
float confidence_threshold = 0.60; // Minimum confidence to display result
int num_classes = 0;

static camera_config_t camera_config = {
    .pin_pwdn = PWDN_GPIO_NUM,
    .pin_reset = RESET_GPIO_NUM,
    .pin_xclk = XCLK_GPIO_NUM,
    .pin_sscb_sda = SIOD_GPIO_NUM,
    .pin_sscb_scl = SIOC_GPIO_NUM,

    .pin_d7 = Y9_GPIO_NUM,
    .pin_d6 = Y8_GPIO_NUM,
    .pin_d5 = Y7_GPIO_NUM,
    .pin_d4 = Y6_GPIO_NUM,
    .pin_d3 = Y5_GPIO_NUM,
    .pin_d2 = Y4_GPIO_NUM,
    .pin_d1 = Y3_GPIO_NUM,
    .pin_d0 = Y2_GPIO_NUM,
    .pin_vsync = VSYNC_GPIO_NUM,
    .pin_href = HREF_GPIO_NUM,
    .pin_pclk = PCLK_GPIO_NUM,

    .xclk_freq_hz = 10000000,
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,

    .pixel_format = PIXFORMAT_JPEG,
    .frame_size = FRAMESIZE_QVGA,

    .jpeg_quality = 12,
    .fb_count = 1,
    .fb_location = CAMERA_FB_IN_PSRAM,
    .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
};

// Function to update camera config with new pins
void update_camera_config(int config_num) {
    if (config_num == 1) {
        camera_config.pin_pwdn = CONFIG_1_PWDN_GPIO_NUM;
        camera_config.pin_reset = CONFIG_1_RESET_GPIO_NUM;
        camera_config.pin_xclk = CONFIG_1_XCLK_GPIO_NUM;
        camera_config.pin_sscb_sda = CONFIG_1_SIOD_GPIO_NUM;
        camera_config.pin_sscb_scl = CONFIG_1_SIOC_GPIO_NUM;
        camera_config.pin_d7 = CONFIG_1_Y9_GPIO_NUM;
        camera_config.pin_d6 = CONFIG_1_Y8_GPIO_NUM;
        camera_config.pin_d5 = CONFIG_1_Y7_GPIO_NUM;
        camera_config.pin_d4 = CONFIG_1_Y6_GPIO_NUM;
        camera_config.pin_d3 = CONFIG_1_Y5_GPIO_NUM;
        camera_config.pin_d2 = CONFIG_1_Y4_GPIO_NUM;
        camera_config.pin_d1 = CONFIG_1_Y3_GPIO_NUM;
        camera_config.pin_d0 = CONFIG_1_Y2_GPIO_NUM;
        camera_config.pin_vsync = CONFIG_1_VSYNC_GPIO_NUM;
        camera_config.pin_href = CONFIG_1_HREF_GPIO_NUM;
        camera_config.pin_pclk = CONFIG_1_PCLK_GPIO_NUM;
    } else if (config_num == 2) {
        camera_config.pin_pwdn = CONFIG_2_PWDN_GPIO_NUM;
        camera_config.pin_reset = CONFIG_2_RESET_GPIO_NUM;
        camera_config.pin_xclk = CONFIG_2_XCLK_GPIO_NUM;
        camera_config.pin_sscb_sda = CONFIG_2_SIOD_GPIO_NUM;
        camera_config.pin_sscb_scl = CONFIG_2_SIOC_GPIO_NUM;
        camera_config.pin_d7 = CONFIG_2_Y9_GPIO_NUM;
        camera_config.pin_d6 = CONFIG_2_Y8_GPIO_NUM;
        camera_config.pin_d5 = CONFIG_2_Y7_GPIO_NUM;
        camera_config.pin_d4 = CONFIG_2_Y6_GPIO_NUM;
        camera_config.pin_d3 = CONFIG_2_Y5_GPIO_NUM;
        camera_config.pin_d2 = CONFIG_2_Y4_GPIO_NUM;
        camera_config.pin_d1 = CONFIG_2_Y3_GPIO_NUM;
        camera_config.pin_d0 = CONFIG_2_Y2_GPIO_NUM;
        camera_config.pin_vsync = CONFIG_2_VSYNC_GPIO_NUM;
        camera_config.pin_href = CONFIG_2_HREF_GPIO_NUM;
        camera_config.pin_pclk = CONFIG_2_PCLK_GPIO_NUM;
    } else if (config_num == 3) {
        camera_config.pin_pwdn = CONFIG_3_PWDN_GPIO_NUM;
        camera_config.pin_reset = CONFIG_3_RESET_GPIO_NUM;
        camera_config.pin_xclk = CONFIG_3_XCLK_GPIO_NUM;
        camera_config.pin_sscb_sda = CONFIG_3_SIOD_GPIO_NUM;
        camera_config.pin_sscb_scl = CONFIG_3_SIOC_GPIO_NUM;
        camera_config.pin_d7 = CONFIG_3_Y9_GPIO_NUM;
        camera_config.pin_d6 = CONFIG_3_Y8_GPIO_NUM;
        camera_config.pin_d5 = CONFIG_3_Y7_GPIO_NUM;
        camera_config.pin_d4 = CONFIG_3_Y6_GPIO_NUM;
        camera_config.pin_d3 = CONFIG_3_Y5_GPIO_NUM;
        camera_config.pin_d2 = CONFIG_3_Y4_GPIO_NUM;
        camera_config.pin_d1 = CONFIG_3_Y3_GPIO_NUM;
        camera_config.pin_d0 = CONFIG_3_Y2_GPIO_NUM;
        camera_config.pin_vsync = CONFIG_3_VSYNC_GPIO_NUM;
        camera_config.pin_href = CONFIG_3_HREF_GPIO_NUM;
        camera_config.pin_pclk = CONFIG_3_PCLK_GPIO_NUM;
    }
}

/* Function definitions ------------------------------------------------------- */
bool ei_camera_init(void);
void ei_camera_deinit(void);
bool ei_camera_capture(uint32_t img_width, uint32_t img_height, uint8_t *out_buf);
void display_results(String class_name, float conf);
String abbreviate_class_name(String full_name);

/**
* @brief      Arduino setup function
*/
void setup()
{
    Serial.begin(115200);
    while (!Serial);
    Serial.println("Generic Edge Impulse Inferencing - XIAO ESP32S3 Sense with OLED");
    
    // Get model information
    num_classes = EI_CLASSIFIER_LABEL_COUNT;
    Serial.print("Model has ");
    Serial.print(num_classes);
    Serial.println(" classes:");
    
    for (int i = 0; i < num_classes; i++) {
        Serial.print("  ");
        Serial.print(i);
        Serial.print(": ");
        Serial.println(ei_classifier_inferencing_categories[i]);
    }
    
    // Initialize OLED display
    u8g2.begin();
    u8g2.clearDisplay();
    
    // Show startup message on OLED
    u8g2.firstPage();
    do {
        u8g2.setFont(u8g2_font_6x10_tr);
        u8g2.setCursor(5, 15);
        u8g2.print("ML Ready");
        u8g2.setCursor(5, 25);
        u8g2.print(String(num_classes) + " classes");
        u8g2.setCursor(5, 35);
        u8g2.print("Starting...");
    } while (u8g2.nextPage());
    
    if (ei_camera_init() == false) {
        ei_printf("Failed to initialize Camera!\r\n");
        u8g2.firstPage();
        do {
            u8g2.setFont(u8g2_font_ncenB08_tr);
            u8g2.setCursor(5, 25);
            u8g2.print("CAM ERR");
        } while (u8g2.nextPage());
    }
    else {
        ei_printf("Camera initialized\r\n");
    }

    ei_printf("\nStarting continuous inference...\n");
    ei_sleep(2000);
}

/**
* @brief      Get data and run inferencing
*/
void loop()
{
    if (ei_sleep(100) != EI_IMPULSE_OK) {
        return;
    }

    snapshot_buf = (uint8_t*)ps_malloc(EI_CAMERA_RAW_FRAME_BUFFER_COLS * EI_CAMERA_RAW_FRAME_BUFFER_ROWS * EI_CAMERA_FRAME_BYTE_SIZE);

    if(snapshot_buf == nullptr) {
        ei_printf("ERR: Failed to allocate snapshot buffer from PSRAM!\n");
        
        snapshot_buf = (uint8_t*)malloc(EI_CAMERA_RAW_FRAME_BUFFER_COLS * EI_CAMERA_RAW_FRAME_BUFFER_ROWS * EI_CAMERA_FRAME_BYTE_SIZE);
        
        if(snapshot_buf == nullptr) {
            ei_printf("ERR: Failed to allocate snapshot buffer from heap too!\n");
            return;
        }
    }

    ei::signal_t signal;
    signal.total_length = EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT;
    signal.get_data = &ei_camera_get_data;

    if (ei_camera_capture((size_t)EI_CLASSIFIER_INPUT_WIDTH, (size_t)EI_CLASSIFIER_INPUT_HEIGHT, snapshot_buf) == false) {
        ei_printf("Failed to capture image\r\n");
        free(snapshot_buf);
        return;
    }

    ei_impulse_result_t result = { 0 };

    EI_IMPULSE_ERROR err = run_classifier(&signal, &result, false);
    
    if (err != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", err);
        free(snapshot_buf);
        ei_sleep(1000);
        return;
    }

    // Print the predictions to serial
    ei_printf("Predictions (DSP: %d ms., Classification: %d ms.): \n",
                result.timing.dsp, result.timing.classification);

    // Find the class with highest confidence
    detected_class = "";
    confidence = 0.0;
    int detected_index = -1;
    
    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        float value = result.classification[i].value;
        ei_printf("  %s: %.5f\r\n", ei_classifier_inferencing_categories[i], value);
        
        if (value > confidence) {
            confidence = value;
            detected_class = String(ei_classifier_inferencing_categories[i]);
            detected_index = i;
        }
    }

    // Always display results (even low confidence shows what's being detected)
    display_results(detected_class, confidence);

#if EI_CLASSIFIER_HAS_ANOMALY
    ei_printf("Anomaly prediction: %.3f\r\n", result.anomaly);
#endif

    free(snapshot_buf);
}

/**
* @brief      Abbreviate class name to fit tiny display
*/
String abbreviate_class_name(String full_name) {
    String abbrev = "";
    full_name.toUpperCase(); // Convert to uppercase for consistency
    
    // Strategy 1: If name is 3 chars or less, use as is
    if (full_name.length() <= 3) {
        return full_name;
    }
    
    // Strategy 2: If name has underscore or dash, use first letter of each part
    if (full_name.indexOf('_') != -1 || full_name.indexOf('-') != -1) {
        int start = 0;
        for (int i = 0; i <= full_name.length(); i++) {
            if (i == full_name.length() || full_name[i] == '_' || full_name[i] == '-') {
                if (start < i && abbrev.length() < 3) {
                    abbrev += full_name[start];
                }
                start = i + 1;
            }
        }
        if (abbrev.length() > 0) return abbrev;
    }
    
    // Strategy 3: Take first 3 consonants (skip vowels for better recognition)
    String consonants = "";
    for (int i = 0; i < full_name.length() && consonants.length() < 3; i++) {
        char c = full_name[i];
        if (c != 'A' && c != 'E' && c != 'I' && c != 'O' && c != 'U') {
            consonants += c;
        }
    }
    if (consonants.length() >= 2) return consonants;
    
    // Strategy 4: Just take first 3 characters
    return full_name.substring(0, 3);
}

/**
* @brief      Display classification results on OLED
*/
void display_results(String class_name, float conf) {
    // Abbreviate the class name
    abbreviated_class = abbreviate_class_name(class_name);
    
    u8g2.firstPage();
    do {
        // Different display based on confidence level
        if (conf > confidence_threshold) {
            // High confidence - large display
            
            // Show abbreviated class name in large font
            u8g2.setFont(u8g2_font_ncenB14_tr);
            
            // Center the abbreviated text
            int text_width = u8g2.getStrWidth(abbreviated_class.c_str());
            int x_pos = (72 - text_width) / 2;
            u8g2.setCursor(x_pos, 18);
            u8g2.print(abbreviated_class);
            
            // Show confidence as percentage in medium font
            u8g2.setFont(u8g2_font_ncenB08_tr);
            String conf_str = String(int(conf * 100)) + "%";
            text_width = u8g2.getStrWidth(conf_str.c_str());
            x_pos = (72 - text_width) / 2;
            u8g2.setCursor(x_pos, 35);
            u8g2.print(conf_str);
            
        } else {
            // Low confidence - show all classes with small font
            u8g2.setFont(u8g2_font_5x7_tr);
            
            // Show "Low Conf" header
            u8g2.setCursor(15, 8);
            u8g2.print("Scanning");
            
            // Show top result even if low confidence
            u8g2.setCursor(5, 20);
            u8g2.print(abbreviated_class + "?");
            
            // Show confidence
            u8g2.setCursor(5, 30);
            u8g2.print(String(int(conf * 100)) + "%");
            
            // Draw scanning animation dots
            static int dot_pos = 0;
            dot_pos = (dot_pos + 1) % 3;
            u8g2.setCursor(45, 30);
            for(int i = 0; i < 3; i++) {
                if(i == dot_pos) u8g2.print(".");
                else u8g2.print(" ");
            }
        }
        
        // Always draw a border
        u8g2.drawFrame(0, 0, 72, 40);
        
    } while (u8g2.nextPage());
}

/**
 * @brief   Setup image sensor & start streaming
 *
 * @retval  false if initialisation failed
 */
bool ei_camera_init(void) {

    if (is_initialised) return true;

    delay(1000);

    // Try Configuration 1 (OV2640 common)
    update_camera_config(1);
    camera_config.xclk_freq_hz = 10000000;
    camera_config.fb_location = CAMERA_FB_IN_PSRAM;
    
    esp_err_t err = esp_camera_init(&camera_config);
    if (err == ESP_OK) {
        goto camera_init_success;
    }

    // Try Configuration 2 (OV3660)
    esp_camera_deinit();
    delay(500);
    update_camera_config(2);
    camera_config.xclk_freq_hz = 16000000;
    camera_config.fb_location = CAMERA_FB_IN_PSRAM;
    
    err = esp_camera_init(&camera_config);
    if (err == ESP_OK) {
        goto camera_init_success;
    }

    // Try Configuration 3 (Mixed)
    esp_camera_deinit();
    delay(500);
    update_camera_config(3);
    camera_config.xclk_freq_hz = 10000000;
    camera_config.fb_location = CAMERA_FB_IN_PSRAM;
    
    err = esp_camera_init(&camera_config);
    if (err == ESP_OK) {
        goto camera_init_success;
    }

    // Try with DRAM instead of PSRAM for all configs
    for (int config = 1; config <= 3; config++) {
        esp_camera_deinit();
        delay(500);
        
        update_camera_config(config);
        camera_config.fb_location = CAMERA_FB_IN_DRAM;
        camera_config.xclk_freq_hz = (config == 2) ? 16000000 : 10000000;
        
        err = esp_camera_init(&camera_config);
        if (err == ESP_OK) {
            goto camera_init_success;
        }
    }

    return false;

camera_init_success:
    sensor_t * s = esp_camera_sensor_get();
    if (s == NULL) {
        return false;
    }
    
    // Apply sensor-specific settings
    if (s->id.PID == OV3660_PID) {
      s->set_vflip(s, 1); 
      s->set_brightness(s, 1); 
      s->set_saturation(s, 0); 
      s->set_hmirror(s, 0); 
    } else if (s->id.PID == 0x26) { // OV2640
      s->set_vflip(s, 1);        
      s->set_hmirror(s, 0);      
    } else {
      s->set_vflip(s, 1);        
      s->set_hmirror(s, 0);      
    }
    
    is_initialised = true;
    return true;
}

/**
 * @brief      Stop streaming of sensor data
 */
void ei_camera_deinit(void) {
    esp_err_t err = esp_camera_deinit();

    if (err != ESP_OK)
    {
        ei_printf("Camera deinit failed\n");
        return;
    }

    is_initialised = false;
    return;
}

/**
 * @brief      Capture, rescale and crop image
 *
 * @param[in]  img_width     width of output image
 * @param[in]  img_height    height of output image
 * @param[in]  out_buf       pointer to store output image
 *
 * @retval     false if not initialised, image captured, rescaled or cropped failed
 *
 */
bool ei_camera_capture(uint32_t img_width, uint32_t img_height, uint8_t *out_buf) {
    bool do_resize = false;

    if (!is_initialised) {
        ei_printf("ERR: Camera is not initialized\r\n");
        return false;
    }

    camera_fb_t *fb = esp_camera_fb_get();

    if (!fb) {
        ei_printf("Camera capture failed\n");
        return false;
    }

   bool converted = fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, snapshot_buf);

   esp_camera_fb_return(fb);

   if(!converted){
       ei_printf("Conversion failed\n");
       return false;
   }

    if ((img_width != EI_CAMERA_RAW_FRAME_BUFFER_COLS)
        || (img_height != EI_CAMERA_RAW_FRAME_BUFFER_ROWS)) {
        do_resize = true;
    }

    if (do_resize) {
        ei::image::processing::crop_and_interpolate_rgb888(
        out_buf,
        EI_CAMERA_RAW_FRAME_BUFFER_COLS,
        EI_CAMERA_RAW_FRAME_BUFFER_ROWS,
        out_buf,
        img_width,
        img_height);
    }

    return true;
}

static int ei_camera_get_data(size_t offset, size_t length, float *out_ptr)
{
    size_t pixel_ix = offset * 3;
    size_t pixels_left = length;
    size_t out_ptr_ix = 0;

    while (pixels_left != 0) {
        // Swap BGR to RGB here
        out_ptr[out_ptr_ix] = (snapshot_buf[pixel_ix + 2] << 16) + (snapshot_buf[pixel_ix + 1] << 8) + snapshot_buf[pixel_ix];

        // go to the next pixel
        out_ptr_ix++;
        pixel_ix+=3;
        pixels_left--;
    }
    return 0;
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_CAMERA
#error "Invalid model for current sensor"
#endif