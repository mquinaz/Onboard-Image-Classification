# Onboard-Image-Classification

##  Basic instructions


### ICA launch

Launch image classification actor (ICA) 
with default parameters (type `./ica.py -h` for options) using the `ica.py` script:

```
$ ./ica.py 
2021-08-05 09:14:02 - INFO - starting
2021-08-05 09:14:02 - INFO - resetting internal state
...
```

### ICA setup

Send `setup` command to ICA using `icmsg.py` (type `./icmsg.py -h` for options):

```
$ ./icmsg.py setup
Message sent to 127.0.0.1:6011 ...
ImageClassificationControl
FFFF:FF -> FFFF:FF
2021/08/05 08:49:25
    command: 2
    model: autoML
    video_source: example.mjpg
    sampling_freq: 1
```

If all goes well, the ICA will be configured and ready for image classification:
  
```
2021-08-05 09:14:06 - INFO - Received control message -- ImageClassificationControl
FFFF:FF -> FFFF:FF
2021/08/05 09:14:06
    command: 2
    model: autoML
    video_source: example.mjpg
    sampling_freq: 1
2021-08-05 09:14:06 - INFO - resetting internal state
2021-08-05 09:14:06 - INFO - setting up
...
2021-08-05 09:14:06 - INFO - setup complete - data dir: /home/edrdo/Onboard-Image-Classification/data/autoML/20210805/091406
```


### Start image classification 

Send `start` command to ICA.

```
$ ./icmsg.py start
Message sent to 127.0.0.1:6011 ...
ImageClassificationControl
FFFF:FF -> FFFF:FF
2021/08/05 08:51:18
    command: 0
    model: 
    video_source: 
    sampling_freq: 0
```

If all goes well, the ICA will start grabbing video frames and classify them:

```
2021-08-05 09:15:43 - INFO - Received control message -- ImageClassificationControl
FFFF:FF -> FFFF:FF
2021/08/05 09:15:43
    command: 0
    model: 
    video_source: 
    sampling_freq: 0
2021-08-05 09:15:43 - INFO - now active
2021-08-05 09:15:43 - INFO - Classification time: 0.099 s
2021-08-05 09:15:43 - INFO - [('A4', 0.4823529411764706), ('A3', 0.4196078431372549), ('A5', 0.10196078431372549)]
2021-08-05 09:15:43 - INFO - /home/edrdo/Onboard-Image-Classification/data/autoML/20210805/091541/f0001.png written!
2021-08-05 09:15:43 - INFO - ImageClassification
FFFF:FF -> FFFF:FF
    frameid: 1
    classifications: [{
      abbrev: ScoredClassification
      score: 48
      classification: A4
    }{
      abbrev: ScoredClassification
      score: 42
      classification: A3
    }{
      abbrev: ScoredClassification
      score: 10
      classification: A5
    }]
    data: 
...
```

### Stop image classification

```
$ ./icmsg.py stop
Message sent to 127.0.0.1:6011 ...
ImageClassificationControl
FFFF:FF -> FFFF:FF
2021/08/05 08:59:53
    command: 1
    model: 
    video_source: 
    sampling_freq: 0
```

The ICA will stop performing image classification. The configuration is not lost though.

```
2021-08-05 09:18:02 - INFO - Received control message -- ImageClassificationControl
FFFF:FF -> FFFF:FF
2021/08/05 09:18:02
    command: 1
    model: 
    video_source: 
    sampling_freq: 0
2021-08-05 09:18:02 - INFO - now inactive
```

## Other utility scripts

### Model testing using `ictest.py`

The `ictest.py` can be used to test classification models (type `ictest.py -h` for options).

```
$ ./ictest.py -m autoML examples/m*.jpg
INFO: Initialized TensorFlow Lite runtime.
--- examples/m3_1.jpg ---
1: A3 93
2: A4 4
3: A5 3
--- examples/m3_2.jpg ---
1: A3 95
2: A4 3
3: A5 2
...
--- examples/m5_2.jpg ---
1: A5 95
2: A3 3
3: A4 3
--- examples/m5_3.jpg ---
1: A5 93
2: A4 4
3: A3 4
```


### Communications testing using `ictest.py`

The `iclisten.py` script can be used to receive and print messages sent by the ICA (type `iclisten.py -h` for options).

```
$ ./iclisten.py
Announce
3335:01 -> FFFF:FF
2021/08/05 09:03:17
    sys_name: ccu-pyimc-rubisco
    sys_type: 0
    owner: 65535
    lat: 0
    lon: 0
    height: 0
    services: imc+udp://10.128.0.4:6012/
...
```

## Detailed usage

### `ica.py` 

```
usage: ica.py [-h] [-i IMC_ADDRESS] [-l LOCAL_PORT] [-d DATA_PATH]
                [-m MODEL_PATH] [-a STATIC_DEST_ADDR] [-p STATIC_DEST_PORT]
                [-w WINDOW_SIZE] [-H]

Image classification actor.

optional arguments:
  -h, --help            show this help message and exit
  -i IMC_ADDRESS, --imc_address IMC_ADDRESS
                        local IMC address for actor
  -l LOCAL_PORT, --local_port LOCAL_PORT
                        local port for incoming messages
  -d DATA_PATH, --data_path DATA_PATH
                        path for generated data files
  -m MODEL_PATH, --model_path MODEL_PATH
                        path for classification models
  -a STATIC_DEST_ADDR, --static_dest_addr STATIC_DEST_ADDR
                        static destination host
  -p STATIC_DEST_PORT, --static_dest_port STATIC_DEST_PORT
                        static destination port
  -w WINDOW_SIZE, --window-size WINDOW_SIZE
                        GUI window size
  -H, --headless        headless mode (no GUI)
```
  
### `icmsg.py`

```
usage: icmsg.py [-h] [-a ADDRESS] [-p PORT] [-m MODEL] [-s SAMPLING_FREQ]
                [-v VIDEO_SOURCE]
                {setup,start,stop}

Send message to image classification actor.

positional arguments:
  {setup,start,stop}    Control command

optional arguments:
  -h, --help            show this help message and exit
  -a ADDRESS, --address ADDRESS
                        host name or IP address
  -p PORT, --port PORT  host port
  -m MODEL, --model MODEL
                        model id (for setup command)
  -s SAMPLING_FREQ, --sampling_freq SAMPLING_FREQ
                        sampling frequency (for setup command)
  -v VIDEO_SOURCE, --video_source VIDEO_SOURCE
                        video source (for setup command)
```
  

### `ictest.py`

```
usage: ictest.py [-h] [-p MODEL_PATH] [-m MODEL] files [files ...]

Test classification models for given images

positional arguments:
  files                 files to classify

optional arguments:
  -h, --help            show this help message and exit
  -p MODEL_PATH, --model_path MODEL_PATH
                        path for classification models
  -m MODEL, --model MODEL
                        model id
```

### `iclisten.py`

```
usage: iclisten.py [-h] [-i IMC_ADDRESS] [-p PORT]

Receive messages from image classification actor.

optional arguments:
  -h, --help            show this help message and exit
  -i IMC_ADDRESS, --imc-address IMC_ADDRESS
                        IMC address
  -p PORT, --port PORT  host port
```


