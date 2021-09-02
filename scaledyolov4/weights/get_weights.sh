# Checks if gdown is installed
if ! type "gdown" > /dev/null; then
    read -p "Install gdown (google drive downloader) with pip3 (y/n)? " yn
    case $yn in
        [Yy]* ) pip3 install gdown;;
        * ) echo "Not installing gdown. Exiting.."; exit;;
    esac
fi

# download weights saved as state_dict
gdown -O yolov4l-mish_-state.pt https://drive.google.com/uc?id=1VIkLSogNsrKdUx3qpxvj-mUKQMtGnifk
# gdown -O yolov4-p5-state.pt https://drive.google.com/uc?id=1J9CLEucAzQnnVucJfAmmhysPyK03A83N
gdown -O yolov4-p5_-state.pt https://drive.google.com/uc?id=1r2rdKxvnXj8mYVF0Ku-U6lAX-4JVllxf
# gdown -O yolov4-p6-state.pt https://drive.google.com/uc?id=1BFTErx0FfAmkTRIhWnI-k0NjqiuCyIsO
gdown -O yolov4-p6_-state.pt https://drive.google.com/uc?id=1lixDNHPhbZ7ZpexIzVPrTo2BEMgCFSIy
gdown -O yolov4-p7-state.pt https://drive.google.com/uc?id=1c8YIBod6AjiJhxLBK0Cjwxfca7eo1_8J
