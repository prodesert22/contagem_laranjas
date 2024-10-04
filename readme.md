# Contagem  de laranjas e pés de erva mate

Esse projeto tem como objetivo fazer a contagem de laranjas e pé de erva mate através de usos de diferentes tecnicas e modelos de inteligência artificial.

## Dataset
O dataset usado para laranjas é uma junção das imagens disponíveis no [kaggle](https://www.kaggle.com/datasets/balraj98/apple2orange-dataset) e [images.cv](https://images.cv/download/orange/1366) assim foi utilizada a plataforma roboflow para fazer as anotações dos rotulos das imagens.

## Técnicas usadas

### Yolo

[YOLO](https://docs.ultralytics.com/#yolo-a-brief-history) (You Only Look Once) é um modelo popular de detecção de objetos e segmentação de imagem. Existem várias versões deste modelo onde abaixo estão listadas as versões usadas.
Com o yolo é necessário fazer o treinamento com base nos modelos abaixo para que ele seja capaz de fazer a classificação.

**Modelos Yolo usados**

- [Yolo v5](https://docs.ultralytics.com/models/yolov5/)
- [Yolo v8](https://docs.ultralytics.com/models/yolov8/)
- [Yolo v9](https://docs.ultralytics.com/models/yolov9/)
- [Yolo v11](https://docs.ultralytics.com/models/yolo11/)