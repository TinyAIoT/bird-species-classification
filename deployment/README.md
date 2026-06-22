# Deployment of: 
# On-Device Bird Species Classification

### Hardware
- senseBox MCU Eye with OV2640 camera
- Strain Gauge Load Cell

### Software
Once the scale measures more than 5 grams 50-200 images are captured and sent out via WiFi to an MQTT Broker. The first of those images is classified and the classification result (classname + confidence) is also sent out.

