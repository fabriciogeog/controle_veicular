"""Controle de acesso veicular com OpenCV."""

#############################################################################

# IMPORTAÇÕES

#############################################################################

import csv
import cv2
import numpy as np
import pandas as pd
import pytesseract
import re
import sqlite3
from datetime import datetime

#############################################################################

# CONSTANTES

#############################################################################

VERMELHO = (0, 0, 255)
VERDE = (0, 255, 0)
AZUL = (255, 0, 0)
BRANCO = (255, 255, 255)
PRETO = (0, 0, 0)
fonte0 = cv2.FONT_HERSHEY_SIMPLEX
fonte1 = cv2.FONT_HERSHEY_PLAIN
fonte2 = cv2.FONT_HERSHEY_DUPLEX
fonte3 = cv2.FONT_HERSHEY_COMPLEX
fonte4 = cv2.FONT_HERSHEY_TRIPLEX
fonte5 = cv2.FONT_HERSHEY_COMPLEX_SMALL
fonte6 = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
fonte7 = cv2.FONT_HERSHEY_SCRIPT_COMPLEX

#############################################################################

# FUNÇÕES

#############################################################################

def yolo_deteccao(quadros):
    net = cv2.dnn.readNetFromONNX('last.onnx')
    file = open('coco.txt', 'r')
    classes = file.read().split('\n')
    blob = cv2.dnn.blobFromImage(quadros, scalefactor=1/255, size=(640, 640), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()[0]

    classes_ids = []
    confidences = []
    boxes = []
    rows = detections.shape[0]

    img_width, img_height = quadros.shape[1], quadros.shape[0]
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


def bbox(quadros, box):
    x = int(box[0])
    y = int(box[1])
    w = int(box[2])
    h = int(box[3])
    roi_placa = quadros[y:y+h, x:x+w]
    cv2.rectangle(quadros, (x, y), (x+w, y+h), BRANCO, 2, 2)
    return roi_placa


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


def consulta_banco(placa_lida):
    """Acessa banco de dados para consulta"""
    conexao = sqlite3.connect('veiculos_cadastrados.db')
    query = f"SELECT * FROM veiculos_cadastrados WHERE PLACA='{placa_lida}'"
    cur = conexao.cursor()
    res = cur.execute(query)
    resultado = res.fetchone()
    return resultado


def gestao_resultado(resultado, placa_conf, quadros):
    """Função que realiza a gestão do resultado."""
    if resultado is not None:
        if placa_conf not in lista_placas:
            lista_placas.append(placa_conf)
            if len(lista_placas) > 30:
                lista_placas.clear()
            cv2.putText(quadros, placa_conf, (25, 150), fonte3, 2, VERDE, 2, cv2.LINE_AA)
            caminho = f"fotos_placas/{placa_conf}{DATA}" + ".jpg"
            foto_placa(caminho, quadros)
            registro_placa(resultado, placa_conf)

    else:
        if placa_conf not in lista_nao_placas:
            lista_nao_placas.append(placa_conf)
            if len(lista_placas) > 30:
                lista_nao_placas.clear()
            cv2.putText(quadros, placa_conf, (25, 150), fonte3, 2, VERMELHO, 2, cv2.LINE_AA)
            caminho = f"fotos_placas_nc/{placa_conf}{DATA}" + ".jpg"
            foto_placa(caminho, quadros)
            registro_placa(resultado, placa_conf)


def qualidade_leitura(placa_lida):
    """Procura realizar a confirmação da leitura da placa."""
    lista = []
    a = 0
    while a < 20:
        lista.append(placa_lida)
        a+=1
    placa_conf = max(lista, key=lista.count)
    return placa_conf


def foto_placa(caminho, quadros):
    """Função para captura da foto da placa lida."""
    cv2.imwrite(caminho, quadros)


def registro_placa(resultado, placa_conf):
    """Função destina-se a realizar o registro dos dados da placa capturada."""
    with open('arquivos_csv/registro_placas.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        if resultado is None:
            writer.writerow(['','',placa_conf,'','', '', DATA])
        else:
            resultado = resultado + (DATA,)
            writer.writerow(resultado)


def consulta_api_denatran():
    """Implementar. A pesquisa será feita na base de dados do DENATRAN."""
    pass


#############################################################################

# APLICAÇÃO PRINCIPAL

#############################################################################

if __name__ == "__main__":

    # LISTAS NECESSÁRIAS (EM BRANCO)
    lista_placas = []
    lista_nao_placas = [None]
    

    # INICIALIZAÇÃO DO VÍDEO E APLICAÇÃO DAS FUNÇÕES
    url = 0 # (0) - webcam embarcada, (2) - webcam conectada
    # url = 'videos_teste/video5.mp4'
    captura = cv2.VideoCapture(url)
    while True:
        DATA = datetime.strftime(datetime.today(), "%d-%m-%Y %H:%M:%S")
        quadros = captura.read()[1]
        # quadros = cv2.resize(quadros, [960, 540]) # Observar a necessidade de realizar o resize""
        cv2.rectangle(quadros, (10,10), (400, 100), AZUL, -1)
        cv2.putText(quadros, 'PA-1', (25, 50), fonte3, 1, BRANCO, 2, cv2.LINE_AA)
        cv2.putText(quadros, DATA, (25, 85), fonte3, 1, BRANCO, 2, cv2.LINE_AA)
        box = yolo_deteccao(quadros)
        if box is None:
            print('Aguardando leitura...')
        else:
            roi_placa = bbox(quadros, box)
            placa_lida = leitura_boxes(roi_placa)
            placa_conf = qualidade_leitura(placa_lida)
            resultado = consulta_banco(placa_conf)
            gestao_resultado(resultado, placa_conf, quadros)
            
        cv2.imshow("Controle de Acesso", quadros)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    captura.release()
    cv2.destroyAllWindows()