# Trabalho Final - NIA - 2026

## Classificação de Imagens de Satélite Utilizando Transfer Learning com MobileNetV2

## Disciplina: Noções de Inteligência Artificial (NIA)

## Ferramentas utilizadas: Python, TensorFlow/Keras, MobileNetV2, Transfer Learning, Early Stopping, Pandas, Seaborn e Matplotlib.

# Objetivo: Desenvolver um sistema de classificação automática de imagens de satélite para identificação de diferentes tipos de cobertura e uso do solo.

#1. Bibliotecas
Bibliotecas necessárias para o desenvolvimento do modelo e suas funções.

#bibliotecas utilizadas durante o desenvolvimento do projeto
import tensorflow as tf
import numpy as np
import datetime

#Download do dataset
import os
import json
from google.colab import userdata

#Visualização
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#metricas de avaliação
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#salvar modelo
from google.colab import drive

# 2. Download do dataset (EuroSAT) e extrair no proprio colab
#- Download automático do dataset diretamente do Kaggle utilizando a API oficial.
#- O arquivo é extraído para o ambiente do Google Colab permitindo acessar as imagens que serão utilizadas no desenvolvimento do modelo.

NOME = userdata.get('NOME_DE_USUARIO')
CHAVE = userdata.get('CHAVE_API')

os.makedirs('/root/.kaggle', exist_ok=True)
with open('/root/.kaggle/kaggle.json', 'w') as f:
    json.dump({"username": NOME, "key": CHAVE}, f)
os.chmod('/root/.kaggle/kaggle.json', 0o600)

!kaggle datasets download -d apollo2506/eurosat-dataset
!unzip -q eurosat-dataset.zip -d dataset_imagens
print("concluido")

# 3. Pré-Processamento e Visualização
#- Principais parâmetros do projeto.
#- Definição do tamanho das imagens e do batch.
#- Divisão do dataset entre treinamento e validação.
#- Visualização e manipulação do dataset.

CAMINHO_DADOS = "dataset_imagens/EuroSAT"
IMG_SIZE = 224 # tamanho que a rede neural MobileNetV2 pede
BATCH_SIZE = 32 #valor escolhido para ter um equilibrio na velocidade de treinamento e consumo de memoria


# aqui separo as imagens em conjuntos de treino e validação e redimensiono todas as imagens para 224x224,
# que é o tamanho esperado pela MobileNetV2.

# utilizei 80% das imagens para treinamento e 20% para validação.


treinamento_data = tf.keras.utils.image_dataset_from_directory(
    CAMINHO_DADOS, validation_split=0.2, subset="training", seed=42,
    image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, label_mode='categorical'
)

val_data = tf.keras.utils.image_dataset_from_directory(
    CAMINHO_DADOS, validation_split=0.2, subset="validation", seed=42,
    image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, label_mode='categorical'
)


# Identifica as classes de cobertura de solo automaticamente pelas pastas
tipos_de_solo = treinamento_data.class_names
OUTPUT_SHAPE = len(tipos_de_solo)
print(f"✅ {OUTPUT_SHAPE} tipos de cobertura de solo identificados:")
print(tipos_de_solo)

# 2.1 Visualizando as imagens

#visualizando algumas imagens antes do treinamento para entender quais padrões a rede neural precisará aprender

plt.figure(figsize=(12,12))
for imagens, labels in treinamento_data.take(1):

    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(imagens[i].numpy().astype("uint8"))
        indice = np.argmax(labels[i])
        plt.title(tipos_de_solo[indice],fontsize=13,fontweight="bold")
        plt.axis("off")

plt.tight_layout()
plt.show()


# 2.2 distribuição das classes

# aqui verifico quantas imagens existem em cada categoria para observar se o dataset está balanceado

quantidade_imagens = []

for classe in tipos_de_solo:
    pasta = os.path.join(CAMINHO_DADOS, classe)
    quantidade_imagens.append(
        len(os.listdir(pasta))
    )

dados = pd.DataFrame({
    "Classe": tipos_de_solo,
    "Quantidade": quantidade_imagens
})

dados

print(f"total de imagens: {sum(quantidade_imagens)}")

# 2.3 quantidade de imagens em cada categoria

plt.figure(figsize=(12,5))

sns.barplot(data=dados,x="Classe",y="Quantidade",palette="Blues_r")
plt.xticks(rotation=45,fontsize=12,fontweight="bold")
plt.yticks(fontsize=12,fontweight="bold")
plt.title("quantidade de imagens por classe",fontsize=16,fontweight="bold")
plt.xlabel("classes",fontsize=13,fontweight="bold")
plt.ylabel("quantidade de imagens",fontsize=13,fontweight="bold")
plt.show()

# 4. Construção do modelo
#- Aplicando a abordagem de Transfer Learning
#- Arquiteteura MobileNetV2 Pré-Treinada
#- Definição das métricas utilizadas durante o treinamento

# carrego a MobileNetV2 já pré-treinada no ImageNet para utilizar e a abordagem de Transfer Learning.

def mobilenet(input_shape=(IMG_SIZE, IMG_SIZE, 3), output_shape=OUTPUT_SHAPE):
    # pega o modelo direto da base do tensorflow
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        pooling='avg',
        weights='imagenet'
    )


    base_model.trainable = False
    modelo = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Dense(units=output_shape, activation="softmax")
    ])

    # Compilando o modelo com as metricas para classificação multiclasse
    modelo.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )
    return modelo

# 5. Treinamento do Modelo
#- Treinamento da Rede Neural com os conjuntos de Treino e Validação
#- Abordagem do Early Stopping

NUM_EPOCHS = 20

# aplicando o early stopping pra ajudar a evitar overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True
)

def treinar_modelo():
    modelo = mobilenet()
    history = modelo.fit(x=treinamento_data, epochs=NUM_EPOCHS,
                        validation_data=val_data,
                        validation_freq=1,
                        callbacks=[early_stopping])

    return modelo, history

# Executa o treino
model, history = treinar_modelo()
print(f"\ntreinamento encerrado em {len(history.history['accuracy'])} epochs")

# 6. Resumo dos resultados obtidos
#- Resultados organizados em tabela
#- Analise geral e visualização do treinamento

# criando um dataframe para visualizar a evolução do treinamento

dados_treinamento = pd.DataFrame({

    "Epoch": range(1, len(history.history["accuracy"]) + 1),
    "Acuracia (treino)": history.history["accuracy"],
    "Acuracia (validação)": history.history["val_accuracy"],
    "Perda (treino)": history.history["loss"],
    "Perda (validação)": history.history["val_loss"]

})



sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(10,3))
ax.axis("off")

tabela = ax.table(cellText=np.round(dados_treinamento.values,3), colLabels=dados_treinamento.columns,loc="center")
tabela.auto_set_font_size(False)
tabela.set_fontsize(12)
tabela.scale(1.3,2.0)

plt.show()

#mostrando os melhores resultados

melhor_epoch = np.argmax(history.history["val_accuracy"])

print(f"Melhor epoch: {melhor_epoch + 1}")

print(f"Acuracia de treinamento: {history.history['accuracy'][melhor_epoch]*100:.2f}%")
print(f"Acuracia de validação: {history.history['val_accuracy'][melhor_epoch]*100:.2f}%")
print(f"Perda de treinamento: {history.history['loss'][melhor_epoch]:.3f}")
print(f"Perda de validação: {history.history['val_loss'][melhor_epoch]:.3f}")

# 7. Curvas de Treinamento (Acurácia VS Perda)
#- Graficos detalhados de acurácia e perda ao longo do treinamento para analise da evolução do treinamento

print("\nGraficos de Desempenho")
acuracia = history.history['accuracy']
val_acuracia = history.history['val_accuracy']
perda = history.history['loss']
val_perda = history.history['val_loss']

plt.figure(figsize=(12,4))

# Acurácia
plt.subplot(1,2,1)
plt.plot(acuracia,linewidth=2.5,label="Treino")
plt.plot(val_acuracia,linewidth=2.5,label="Validação")
plt.title("Acurácia",fontsize=16,fontweight="bold")
plt.xlabel("Epoch",fontsize=12,fontweight="bold")
plt.ylabel("Accuracy",fontsize=12,fontweight="bold")
plt.xticks(fontsize=11,fontweight="bold")
plt.yticks(fontsize=11,fontweight="bold")
plt.grid(alpha=0.4)
plt.legend(fontsize=11)


# Perda
plt.subplot(1,2,2)
plt.plot(perda,linewidth=2.5,label="Treino")
plt.plot(val_perda,linewidth=2.5,label="Validação")
plt.title("Perda",fontsize=16,fontweight="bold")
plt.xlabel("Epoch",fontsize=12,fontweight="bold")
plt.ylabel("Loss",fontsize=12,fontweight="bold")
plt.xticks(fontsize=11,fontweight="bold")
plt.yticks(fontsize=11,fontweight="bold")
plt.grid(alpha=0.4)
plt.legend(fontsize=11)

plt.tight_layout()
plt.show()

# 8. Validação visual das previsões
#- Analise visual com algumas imagens do conjunto de validação e suas classes reais
#- Classe Prevista e nível de confiança de cada previsão

# Pega o lote de 32 imagens e labels do dataset de validação e faz o modelo adivinhar quais imagens são.
val_images, val_labels = next(iter(val_data))

preds = model.predict(val_images)

plt.figure(figsize=(15,12))

# determinei 9 imagens mas pode ser mais.
for i in range(9):

    plt.subplot(3,3,i+1)

    # converte as imagens que estão em float pra inteiros
    imagem = val_images[i].numpy().astype("uint8")
    plt.imshow(imagem)

    # olha qual o índice com maior porcentagem
    pred_index = np.argmax(preds[i])
    true_index = np.argmax(val_labels[i])

    # Pega o nome da classe
    pred_label = tipos_de_solo[pred_index]
    true_label = tipos_de_solo[true_index]
    confidence = np.max(preds[i]) * 100

    # acertou = verde, errou = vermelho
    cor = "green" if pred_index == true_index else "red"

    plt.title(
        f"Previsão: {pred_label}\n"
        f"Real: {true_label}\n"
        f"Confiança: {confidence:.1f}%",color=cor,fontsize=11,fontweight="bold")

    plt.axis("off")

plt.suptitle("Validação visual das previsões",fontsize=18,fontweight="bold")
plt.tight_layout()
plt.show()

# 9. Matrizes de confusão
#- Matriz para análise de quantidade de acertos e erros realizados pelo modelo
#- Identificação das classes que foram corretamente classificadas e das principais confusões entre categorias semelhantes
#- Matriz de confusão normalizada em porcentagem para comparação de desempenho entre as diferentes classes

# Quais categorias a rede neural conseguiu classificar corretamente e quais apresentaram maior quantidade de erros.


# listas que armazena os valores reais e previstos

y_real = []
y_previsto = []

# for pra varrer todo o conjunto de validação

for imagens, labels in val_data:

    previsoes = model.predict(imagens, verbose=0)

    y_real.extend(np.argmax(labels.numpy(), axis=1))

    y_previsto.extend(np.argmax(previsoes, axis=1))

# matriz

matriz = confusion_matrix(
    y_real,
    y_previsto
)


plt.figure(figsize=(11,9))

sns.heatmap(matriz,annot=True,fmt="d",cmap="Blues",annot_kws={"size":11,"weight":"bold"},xticklabels=tipos_de_solo,yticklabels=tipos_de_solo)

plt.xticks(rotation=45,ha="right",fontsize=11,fontweight="bold")
plt.yticks(rotation=0,fontsize=11,fontweight="bold")

plt.xlabel("Classe prevista",fontsize=13,fontweight="bold")

plt.ylabel("Classe real",fontsize=13,fontweight="bold")

plt.xlabel("Classe prevista")
plt.ylabel("Classe real")
plt.title("Matriz de confusão",fontsize=18,fontweight="bold")
plt.show()

#Matriz de confusão normalizada (porcentagem)

matriz_normalizada = matriz.astype("float")
matriz_normalizada = (matriz_normalizada /matriz_normalizada.sum(axis=1)[:, np.newaxis])

plt.figure(figsize=(11,9))

sns.heatmap(matriz_normalizada, annot=True,fmt=".2f",cmap="Blues",annot_kws={"size":11, "weight":"bold"}, xticklabels=tipos_de_solo,yticklabels=tipos_de_solo)

plt.xticks(rotation=45,ha="right",fontsize=11,fontweight="bold")
plt.yticks(rotation=0,fontsize=11,fontweight="bold")

plt.xlabel("Classe prevista",fontsize=13, fontweight="bold")
plt.ylabel("Classe real", fontsize=13,fontweight="bold")

plt.title("Matriz de confusão normalizada",fontsize=18, fontweight="bold")

plt.tight_layout()
plt.show()

# 10. Relatório das métricas
#- Avaliação de desempenho de cada classe
#- Principais Metricas de avaliação do modelo
#- Complementa a análise da matriz de confusão.

#alem da acurácia geral, aqui utilizo outras metricas para avaliar o desempenho individual de cada categoria.

relatorio = classification_report(y_real,y_previsto,target_names=tipos_de_solo,output_dict=True)

relatorio = pd.DataFrame(relatorio).transpose()

plt.figure(figsize=(8,6))

sns.heatmap(relatorio.iloc[:-3, :-1],annot=True,cmap="Blues",fmt=".2f",linewidths=.5,annot_kws={
        "fontsize":11,
        "fontweight":"bold"
    })

plt.title("Relatório de classificação",fontsize=18,fontweight="bold")

plt.xlabel("Métricas",fontsize=13,fontweight="bold")

plt.ylabel("Classes",fontsize=13,fontweight="bold")

plt.xticks(fontsize=11,fontweight="bold")
plt.yticks(fontsize=11,fontweight="bold")

plt.tight_layout()
plt.show()

# 11. Salvando e carregando o modelo

# salvando o modelo para reutilizar sem precisar treinar o modelo outra vez

drive.mount('/content/drive')


caminho_modelo = "/content/drive/MyDrive/modelo_eurosat.keras"

model.save(caminho_modelo)

print(f"salvo em: {caminho_modelo}")

# carregando um modelo salvo

modelo_carregado = tf.keras.models.load_model(

    "/content/drive/MyDrive/modelo_eurosat.keras"

)
