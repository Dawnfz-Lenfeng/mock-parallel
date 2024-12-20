from torchsummary import summary

from src.models.yolo import myYOLO

if __name__ == "__main__":
    IMSIZE = 224
    my_YOLO = myYOLO()

    summary(my_YOLO, [(3, IMSIZE, IMSIZE)])
