"""Controle de acesso veicular com OpenCV."""

#  Importações

import csv
import cv2
import numpy as np
import pandas as pd
import pytesseract
import re
import sqlite3
from datetime import datetime


# Constantes

DATA = datetime.strftime(datetime.today(), "%d/%m/%Y %H:%M:%S")


# Funções

def yolo_deteccao(frames):
    net = cv2.dnn.readNetFromONNX('last.onnx')
    file = open('coco.txt', 'r')
    classes = file.read().split('\n')
    blob = cv2.dnn.blobFromImage(frames, scalefactor=1/255, size=(640, 640), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()[0]

    classes_ids = []
    confidences = []
    boxes = []
    rows = detections.shape[0]

    img_width, img_height = frames.shape[1], frames.shape[0]
    x_scale = img_width/640
    y_scale = img_height/640

    for i in range(rows):
        row = detections[i]
        confidence = row[4]
        if confidence > 0.5:
            classes_score = row[5:]
            ind = np.argmax(classes_score)
            if classes_score[ind] > 0.5:
                classes_ids.append(ind)
                confidences.append(confidence)
                cx, cy, w, h = row[:4]
                x1 = int((cx- w/2)*x_scale)
                y1 = int((cy-h/2)*y_scale)
                width = int(w * x_scale)
                height = int(h * y_scale)
                box = np.array([x1,y1,width,height])
                boxes.append(box)
            return box


def foto_placa(caminho, placa_cinza):
    """Registra a foto da placa."""
    foto = np.array(placa_cinza)
    path = caminho
    cv2.imwrite(path, foto)


def registro_placas(resultado):
    """Registro textual das placas capturadas"""
    with open('arquivos_csv/registro_placas.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(resultado)


def leitura_boxes(roi_placa):
    """Leitura das placas na imagem (boxes)."""
    placa_lida = ""
    custom_config = (
        r"-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6"
    )
    boxes = pytesseract.image_to_boxes(roi_placa, config=custom_config)
    for b in boxes.splitlines():
        b = b.split(" ")
        placa_lida = placa_lida + b[0]
    placa_lida = placa_lida.replace("-", "")
    padrao = r"[A-Z]{3}\d{1}[A-Z0-9]{1}\d{2}$"
    match = re.match(padrao, placa_lida)
    if not match:
        pass
    else:
        return placa_lida


def aplica_texto(placa_cinza, placa_lida):
    """Aplica texto aos frames exibidos."""
    cv2.putText(
        placa_cinza,
        "Controle de Acesso",
        (50, 50),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        placa_cinza,
        placa_lida,
        (50, 140),
        cv2.FONT_HERSHEY_COMPLEX,
        2,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        placa_cinza,
        DATA,
        (50, 80),
        cv2.FONT_HERSHEY_COMPLEX,
        0.8,
        (255, 255, 255),
        1,
    )


def controla_registros(resultado, placa_lida, placa_cinza):
    """Controla os registros das placas cadastradas."""
    while resultado is not None:
        DATA = datetime.today()
        if placa_lida not in lista_placas:
            caminho = f"fotos_placas/{placa_lida}{DATA}" + ".jpg"
            foto_placa(caminho, placa_cinza)
            registro_placas(resultado)
            lista_placas.append(placa_lida)
            if len(lista_placas) > 30:
                lista_placas.clear()
            else:
                pass
            print(f"Placas cadastradas: {lista_placas}")
        break


def anota_placas(placa_lida):
    """Realiza a inclusão em lista de placas não cadastradas."""
    with open(
        "arquivos_csv/veiculos_nao_cadastrados.csv", mode="a", newline=""
    ) as cadastro:
        registrar = csv.writer(
            cadastro, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        registrar.writerow([placa_lida, DATA])



def consulta_banco(conexao, query, placa_lida):
    """Acessa banco de dados para consulta"""
    DATA0 = datetime.today()
    cur = conexao.cursor()
    res = cur.execute(query)
    resultado = res.fetchone()
    if resultado is None:
        if placa_lida not in lista_nao_placas:
            caminho = f"fotos_placas_nc/{placa_lida}{DATA0}" + ".jpg"
            foto_placa(caminho, placa_cinza)
            anota_placas(placa_lida)
            lista_nao_placas.append(placa_lida)
            print(f"Placas não cadastradas: {lista_nao_placas}")
        else:
            pass
    else:
        return resultado



# Aplicação principal

if __name__ == "__main__":

    lista_placas = []
    lista_nao_placas = [None]
    conexao = sqlite3.connect('veiculos_cadastrados.db')

    url = 0
    # url = 'http://192.168.1.75:8080/video'
    # url = 'videos_teste/video11.mp4'

    captura = cv2.VideoCapture(url)
    while True:
        quadros = captura.read()[1]
        # quadros = cv2.resize(quadros, [960, 540])
        placa_cinza = cv2.cvtColor(quadros, cv2.COLOR_BGR2GRAY)
        box = yolo_deteccao(quadros)

        if box is None:
            pass
        else:

            print(box)
            x = int(box[0])
            y = int(box[1])
            w = int(box[2])
            h = int(box[3])
            roi_placa = placa_cinza[y:y+h, x:x+w]
            placa_lida = leitura_boxes(roi_placa)
            query = f"SELECT * FROM veiculos_cadastrados WHERE PLACA='{placa_lida}'"
            aplica_texto(placa_cinza, placa_lida)
            resultado = consulta_banco(conexao, query, placa_lida)
            if resultado is None:
                pass
            else:
                print(resultado[0])
            controla_registros(resultado, placa_lida, placa_cinza)
            cv2.rectangle(placa_cinza, (x, y), (x+w, y+h), (255, 255, 255), 2)
            # cv2.imshow("Registro anterior", roi_placa)

        cv2.imshow("Controle de Acesso", placa_cinza)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    captura.release()
    cv2.destroyAllWindows()
