### ResNet50

- base + image + text = 2g for 64 batch size
- base + image + text + audio = 4g for 64 batch size
- base + image + text + audio + video = 8g for 64 batch size

- centi + image + text = 1g for 64 batch size (can probably fit 256)
- centi + image + text + audio = 1g for 64 batch size
- centi + image + text + audio + video = 4g for 64 batch size (can probably fit 96+)

### ViTransformer16

- base + image + text = 2g for 64 batch size
- base + image + text + audio = 4g for 64 batch size
- base + image + text + audio + video = 8g for 64 batch size

- centi + image + text = 1g for 64 batch size (can probably fit 256)
- centi + image + text + audio = 1g for 64 batch size <-
- centi + image + text + audio + video = 4g for 64 batch size (can probably fit 96+)


````
/mnt/disk/tali/experiments//TALI-gcp-sweep-1-milli-tali-centi_modus_prime_resnet50-video-False-audio-True-text-True-image-True-20221601//checkpoints/
````
