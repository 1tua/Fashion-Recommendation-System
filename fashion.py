import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Model
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import GlobalMaxPooling2D
import cv2
import cv2 as cv
from PIL import Image
import pickle
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import pickle

def clean_df(data):
    data.drop(columns=['Unnamed: 5','Unnamed: 6','Unnamed: 7'],inplace=True)

def translate(data):
    # Translating italian micro category into english 
    data=data.replace(to_replace =['GONNA PELLE', 'PANTALONE PELLE', 'CAPOSPALLA PELLE',
           'CAPISPALLA PELLICCIA', 'GILET', 'ABITO', 'COLLI', 'CINTURA',
           'SCIARPA', 'SCIALLE', 'OCCHIALI', 'PORTACHIAVI', 'CAPPELLO',
           'PORTAMONETE', 'CRAVATTA', 'PORTAFOGLIO', 'GUANTI', 'FOULARD',
           'STOLA', 'Accessori per capelli', 'PORTADOCUMENTI', 'Bustina',
           'MANICHE', 'FIBBIE', 'BRETELLE', 'PAPILLON', 'PORTA-AGENDA',
           'OMBRELLO', 'Taccuini', 'Agende', 'BORSA GRANDE IN TESSUTO',
           'BORSA PICCOLA IN PELLE', 'BORSA GRANDE IN PELLE',
           'BORSA MEDIA IN TESSUTO', 'BORSA MEDIA IN PELLE', 'POCHETTE',
           'BORSA PICCOLA IN TESSUTO', 'MARSUPIO', 'ZAINO', 'ZEPPE',
           'BALLERINA', 'SNEAKERS', 'STIVALETTI', 'SANDALI CON TACCO',
           'DECOLLETES', 'SANDALI', 'MOCASSINI ', 'DECOLLETES CON PLATEAU',
           'INFRADITO', 'SANDALI CON PLATEAU', 'DECOLLETES OPEN TOE',
           'STRINGATA', 'STIVALI', 'CIABATTE', 'Peep-toe ballet flats',
           'Decolletes slingback', 'SNEAKER ALTA', 'BABBUCCE',
           'STIVALI CON TACCO', 'SNEAKER SLIP ON', 'MOCASSINI CON TACCO',
           'SABOT', 'POLACCHINA', 'SHOE BOOTS', 'ANFIBI', 'PANTOFOLE'],  
                                value =['LEATHER SKIRT', 'LEATHER TROUSERS', 'LEATHER OUTERWEAR',
    'FUR COATS', 'VESTS', 'DRESS', 'COLLI', 'BELT',
    'SCARF', 'SHAWL', 'GLASSES', 'KEYCHAIN', 'HAT',
    'PURSE', 'TIE', 'WALLET', 'GLOVES', 'FOULARD',
    'STOLA', 'Hair accessories', 'DOCUMENT HOLDER', 'Sachet',
    'SLEEVES', 'BUCKLES', 'BRACES', 'PAPILLON', 'BOOK-HOLDER',
    'UMBRELLA', 'Notebooks', 'Agendas', 'LARGE FABRIC BAG',
    'SMALL LEATHER BAG', 'LARGE LEATHER BAG',
    'MEDIUM FABRIC BAG', 'MEDIUM LEATHER BAG', 'CLUTCH',
    'SMALL FABRIC BAG', 'BABY BAG', 'BACKPACK', 'WEDGES',
    'BALLERINA', 'SNEAKERS', 'ANKLE BOOTS', 'SANDALS WITH HEEL',
    'DECOLLETES', 'SANDALS', 'LOAFERS', 'DECOLLETES WITH PLATEAU',
    'FLIP FLOPS', 'SANDALS WITH PLATEAU', 'DECOLLETES OPEN TOE',
    'LACE UP', 'BOOTS', 'SLIPPERS', 'Peep-toe ballet flats',
    'Decolletes slingback', 'HIGH SNEAKER', 'BABBUCCE',
    'BOOTS WITH HEEL', 'SNEAKER SLIP ON', 'LOAFERS WITH HEEL',
    'SABOT', 'POLACCHINA', 'SHOE BOOTS', 'ANFIBI', 'SLIPPERS'])
    
def translate_colours(data):
    data=data.replace(to_replace =['BRONZO', 'PLATINO', 'NOCCIOLA', 'ORO', 'ANTRACITE', 'NERO',
           'GRIGIO CHIARO', 'COLONIALE', 'BORDEAUX', 'VERDE PETROLIO',
           'CAMMELLO', 'TORTORA', 'MARRONE', 'GIALLO CHIARO', 'RUGGINE',
           'CORALLO', 'CACAO', 'AVIO', 'ARGENTO', 'TURCHESE', 'VERDE SCURO',
           'VIOLA', 'ROSA', 'GIALLO', 'CUOIO', 'MELANZANA', 'AVORIO', 'ROSSO',
           'GRIGIO', 'BEIGE', 'TESTA DI MORO', 'RAME', 'ALBICOCCA', 'SALMONE',
           'VERDE SMERALDO', 'OCRA', 'CARNE', 'MATTONE', 'VIOLA CHIARO',
           'VIOLA SCURO', 'BLU CHINA', 'MALVA', 'ROSA ANTICO', 'VERDE ACIDO',
           'CELESTE', 'ARANCIONE', 'VERDE CHIARO', 'VERDE', 'SABBIA', 'LILLA',
           'AZZURRO', 'VERDE MILITARE', 'PIOMBO', 'CARTA DA ZUCCHERO',
           'BLU SCURO', 'FUCSIA', 'BLU', 'PORPORA', 'BIANCO', 'TRASPARENTE',
           'ROSA CHIARO'],  
                                value =['BRONZE', 'PLATINUM', 'HAZELNUT', 'GOLD', 'ANTHRACITE', 'BLACK',
    'LIGHT GRAY', 'COLONIAL', 'BORDEAUX', 'GREEN PETROLEUM',
    'CAMEL', 'TAUPE', 'BROWN', 'LIGHT YELLOW', 'RUST',
    'CORAL', 'COCOA', 'AVIO', 'SILVER', 'TURQUOISE', 'DARK GREEN',
    'PURPLE', 'PINK', 'YELLOW', 'LEATHER', 'EGGPLANT', 'IVORY', 'RED',
    'GRAY', 'BEIGE', 'DARK BROWN', 'COPPER', 'APRICOT', 'SALMON',
    'GREEN EMERALD', 'OCHER', 'MEAT', 'BRICK', 'LIGHT PURPLE',
    'DARK PURPLE', 'BLU CHINA', 'MALVA', 'ANTIQUE PINK', 'ACID GREEN',
    'LIGHT BLUE', 'ORANGE', 'LIGHT GREEN', 'GREEN', 'SAND', 'LILAC',
    'LIGHT BLUE', 'MILITARY GREEN', 'LEAD', 'SUGAR PAPER',
    'DARK BLUE', 'FUCHSIA', 'BLUE', 'PURPLE', 'WHITE', 'TRANSPARENT',
    'LIGHT PINK'])

def rename_image(data):
    data.filename=data['filename'].apply(lambda x: x[:-4])
    data.filename=data['filename'].apply(lambda x: x + "_resized.jpg")


def plot_bargraph(data, figsize=(15,12)):
    # Plot bar graph of all mirco categories in dataset
    fig= plt.figure(figsize=figsize)
    data.micro_category.value_counts(ascending=True).plot(kind='barh')
    plt.xlabel('Count')
    plt.ylabel('Category')


def plot_figures(figures, nrows = 1, ncols=1,figsize=(8, 8)):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows,figsize=figsize)
    for ind,title in enumerate(figures):
        axeslist.ravel()[ind].imshow(cv2.cvtColor(figures[title], cv2.COLOR_BGR2RGB))
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout() # optional
    
def img_path(img):
    Data_Path= '/Users/flatironschool/Downloads/part1/'
    return Data_Path+img

def resnet_model(input_shape=(197,197,3)):
    image_path = '/Users/flatironschool/Downloads/part1/35154736FEF_resized.jpg'
    im = Image.open(image_path)
    width, height = im.size
    # Pre-Trained Model
    base_model = ResNet50(weights='imagenet', 
                          include_top=False, 
                          input_shape = input_shape)
    base_model.trainable = False

    # Add Layer Embedding
    model = models.Sequential([
        base_model,
        GlobalMaxPooling2D()
    ])
    model.summary()
    pickle.dump(model, open("model.pkl", "wb"))

def load_image(img):
    return cv2.imread(img_path(img))

def get_recommender(idx, data, top_n = 6):
    indices = pd.Series(range(len(data)), index=data.index)
    sim_idx    = indices[idx]
    cosine_sim = pickle.load(open("cosine.pkl", "rb"))
    sim_scores = list(enumerate(cosine_sim[sim_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    idx_rec    = [i[0] for i in sim_scores]
    idx_sim    = [i[1] for i in sim_scores]
    return indices.iloc[idx_rec].index, idx_sim


def get_embedding(model, img_name):
    # Reshape
    img = image.load_img(img_path(img_name), target_size=(197, 197))
    # img to Array
    x   = image.img_to_array(img)
    # Expand Dim (1, w, h)
    x   = np.expand_dims(x, axis=0)
    # Pre process Input
    x   = preprocess_input(x)
    return model.predict(x).reshape(-1)

def rec_top6(idx_ref):
    # Idx Item to Recommender
    # Recommendations
    data = pickle.load(open("data.pkl", "rb"))

    idx_rec, idx_sim = get_recommender(idx_ref, data, top_n = 6)
    # Plot
    plt.imshow(cv2.cvtColor(load_image(data.iloc[idx_ref].filename), cv2.COLOR_BGR2RGB))
    # generation of a dictionary of (title, images)
    figures = {'image'+str(i): load_image(row.filename) for i, row in data.loc[idx_rec].iterrows()}
    # plot of the images in a figure, with 2 rows and 3 columns
    plot_figures(figures, 2, 3)

def plot_tsne():
    tsne = TSNE(n_components=2, verbose=0, perplexity=30, n_iter=12000, init='pca')
    tsne_results = tsne.fit_transform(data_embs)
    data['tsne-2d-one'] = tsne_results[:,0]
    data['tsne-2d-two'] = tsne_results[:,1]
    plt.figure(figsize=(16,10))
    sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two",
                hue="macro_category(english)",
                data=data4,
                legend="full",
                palette="RdBu",
                alpha=0.8)