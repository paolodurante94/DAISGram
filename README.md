# DAISGram
Software per elaborazione immagini in formato bitmap attraverso l'applicazione di filtri, convoluzioni e altre operazioni tra tensori. Sviluppato in C++.

# Risultati Attesi
La cartella `images` contiene alcune immagini con cui provare l'implementazione.

Nella cartella `results` sono presenti i risultati attesi.

Immagine | Brighten (+20) | Brighten (+100) | Grayscale
------------ | ------------- | ------------- |-------------
![DAIS](https://github.com/xwasco/DAISGram_20_21/blob/main/images/dais.bmp) | ![DAIS+20](https://github.com/xwasco/DAISGram_20_21/blob/main/results/dais_brighten_20.bmp) | ![DAIS+100](https://github.com/xwasco/DAISGram_20_21/blob/main/results/dais_brighten_100.bmp) | ![DAIS+100](https://github.com/xwasco/DAISGram_20_21/blob/main/results/dais_gray.bmp) 

Immagine | Smooth (h=3) | Smooth (h=5) | Smooth (h=7)
------------ | ------------- | ------------- | ------------- 
![DAIS](https://github.com/xwasco/DAISGram_20_21/blob/main/images/dais.bmp) | ![DAIS+100](https://github.com/xwasco/DAISGram_20_21/blob/main/results/dais_smooth_3.bmp) | ![DAIS+100](https://github.com/xwasco/DAISGram_20_21/blob/main/results/dais_smooth_5.bmp) | ![DAIS+100](https://github.com/xwasco/DAISGram_20_21/blob/main/results/dais_smooth_7.bmp) 

Immagine | Sharp | Edge | Warhol
------------ | ------------- | ------------- | ------------- 
![DAIS](https://github.com/xwasco/DAISGram_20_21/blob/main/images/dais.bmp) | ![DAIS+20](https://github.com/xwasco/DAISGram_20_21/blob/main/results/dais_sharp.bmp) | ![DAIS+100](https://github.com/xwasco/DAISGram_20_21/blob/main/results/dais_edge.bmp) | ![DAIS+100](https://github.com/xwasco/DAISGram_20_21/blob/main/results/dais_warhol.bmp) | 

Immagine A | Immagine B | Blend alpha=0 | alpha=0.25 | alpha=0.5 | alpha=0.75 | alpha=1.00
------------ | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- 
![DAIS](https://github.com/xwasco/DAISGram_20_21/blob/main/images/blend/blend_a.bmp) | ![DAIS](https://github.com/xwasco/DAISGram_20_21/blob/main/images/blend/blend_b.bmp) | ![DAIS](https://github.com/xwasco/DAISGram_20_21/blob/main/results/blend/blend_0.00.bmp) | ![DAIS](https://github.com/xwasco/DAISGram_20_21/blob/main/results/blend/blend_0.25.bmp) | ![DAIS](https://github.com/xwasco/DAISGram_20_21/blob/main/results/blend/blend_0.50.bmp) | ![DAIS](https://github.com/xwasco/DAISGram_20_21/blob/main/results/blend/blend_0.75.bmp) | ![DAIS](https://github.com/xwasco/DAISGram_20_21/blob/main/results/blend/blend_1.00.bmp) 

# Parti Opzionali
## Green Screen
Primo Piano | Sfondo | GreenScreen
------------ | ------------- | -------------
![DAIS](https://github.com/xwasco/DAISGram_20_21/blob/main/images/greenscreen/gs_2.bmp) | ![DAIS+20](https://github.com/xwasco/DAISGram_20_21/blob/main/images/greenscreen/gs_2_bkg.bmp) | ![DAIS+100](https://github.com/xwasco/DAISGram_20_21/blob/main/results/greenscreen/dais_matrix.bmp)
 | | | RGB={144, 208, 49}, threshold={100, 100, 50}
 |  |  | 
![DAIS](https://github.com/xwasco/DAISGram_20_21/blob/main/images/greenscreen/gs_4.bmp) | ![DAIS+20](https://github.com/xwasco/DAISGram_20_21/blob/main/images/greenscreen/gs_4_bkg.bmp) | ![DAIS+100](https://github.com/xwasco/DAISGram_20_21/blob/main/results/greenscreen/seba_flower.bmp)
 | | | RGB={226,225,220}, threshold={50,50,50}
 
 ## Equalizzazione dell'istogramma
 Immagine Originale | Equalizzata
------------ | -------------
![DAIS](https://github.com/xwasco/DAISGram_20_21/blob/main/images/fullmoon.bmp) | ![DAIS+20](https://github.com/xwasco/DAISGram_20_21/blob/main/results/fullmoon_equalize.bmp)
