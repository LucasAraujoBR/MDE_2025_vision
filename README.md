
# 📚 Detector de Símbolos Matemáticos com YOLOv8

Projeto realizado como atividade prática da disciplina de **Visão Computacional**, com o objetivo de treinar um modelo YOLO para detectar símbolos matemáticos em imagens (números, sinal de mais e de igual).

---

## 🧠 Objetivo

Treinar um modelo YOLOv8 com imagens contendo expressões como `4 + 5 = 9`, detectando:

- Dígitos (0–9)
- Sinal de adição (`+`)
- Sinal de igualdade (`=`)

---

## 📁 Estrutura do Projeto

```
viz_yolo_mde_2025/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
├── utils/
│   └── generate_notes.py
├── runs/
│   └── detect/
│       └── train/
│           ├── weights/
│           │   └── best.pt
│           └── results.png
```

---

## 🧪 Etapas

### ✅ 1. Coleta de Imagens
Imagens PNG contendo expressões matemáticas simples escritas à mão foram organizadas nas pastas `images/train/` e `images/val/`.

### ✅ 2. Anotação Automática com `pytesseract`
Foi utilizado um script Python (`utils/generate_notes.py`) com a biblioteca `pytesseract` para realizar OCR nas imagens e gerar os arquivos de anotação no formato YOLO.

> O script inclui **pré-processamento das imagens** para melhorar a acurácia do reconhecimento óptico.

### ✅ 3. Formato YOLO
As anotações foram salvas em arquivos `.txt`, um por imagem, com a seguinte estrutura:

```
<class_id> <x_center> <y_center> <width> <height>
```

Todos os valores são normalizados (entre 0 e 1).  
Classes utilizadas:
- `0`: dígito (0–9)
- `1`: sinal de adição (+)
- `2`: sinal de igualdade (=)

### ✅ 4. Treinamento com Ultralytics YOLOv8

**Comando utilizado:**
```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

Modelo final salvo em:

```
runs/detect/train/weights/best.pt
```

---

## 📊 Resultados

### 🔧 Métricas Finais do Modelo (época 50):

| Métrica                     | Valor          |
|----------------------------|----------------|
| Precision                  | **0.19893**     |
| Recall                     | **0.37778**     |
| mAP@0.5                    | **0.24193**     |
| mAP@0.5:0.95               | **0.19312**     |
| Val box loss               | **0.74297**     |
| Val dfl loss               | **0.92454**     |

📈 Gráfico de evolução salvo em `results.png`.

---

## 🧪 Inferência

**Comando utilizado:**
```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=images/val
```

---

## 📂 Arquivo `data.yaml`

```yaml
train: images/train
val: images/val

nc: 3
names: ['digit', 'plus', 'equal']
```

---

## ✅ Entregáveis

- [x] Dataset anotado automaticamente com `pytesseract` no formato YOLO
- [x] Arquivo `data.yaml`
- [x] Modelo final treinado: `best.pt`
- [x] Print da inferência com a equação reconhecida

---

## 👩‍🏫 Referência da Aula

Baseado nos slides da disciplina de **Visão Computacional** (UFRPE), adaptado para uso com OCR automático em vez de anotação manual via LabelImg.

---

## 💻 Requisitos

- Python 3.12
- pacotes: `ultralytics`, `pytesseract`, `opencv-python`, `numpy`, `Pillow`

---

## 🔧 Execução do Script de Anotação

```bash
python utils/generate_notes.py
```

Esse script processa as imagens de `images/train/` e `images/val/`, aplica OCR com `pytesseract`, e gera as anotações em `labels/train/` e `labels/val/`.

---
