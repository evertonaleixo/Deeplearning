# Deep Learning -- Redes Convolucionais

Inteligencia artificial é uma área que vem crescendo bastante, principalmente depois de 2012 com a disceminação das deep learnings. Isso foi possível, principalmente pelo maior poder de processamento dos hardwares (CPUs e GPUs) e a grande quantidade de dados disponíveis, herdados do big data. 

Essa nova geração de inteligência artificial (IA) é considerada a segunda. Ela é caracterizada pelo forte uso das propriedade da estatística para melhorar a acurácia de seus modelos. Dentro da área de IA temos as redes neurais. As redes neurais são modelos de AI que imitam o cerebro e são utilizadas para resolver diversos problemas, como reconhecimento e segmentação em imagens, predições textuais, reconhecimento de voz, entre outros. Com a acenssão das redes profundas (deep networks) elas começaram a ter sucesso em instâncias mais complexas.

As redes neurais são um conjunto de camadas conectadas que recebe uma entrada, processa por todas as camadas e gera uma saída. Existem diversos tipos de camadas e para cada problema deve-se utilizar a camada mais apropriada. Para o problema de reconhecimento e segmentação de objetos em imagens, as redes convolucionais tem apresentado ótimos resultados.

As redes convolucionais são uma especialização das redes neurais. Essas redes apresentam quatro tipos de camadas, sendo uma delas um classificador, também chamado de camada fully-connected. A arquitetura de uma rede convolucional é formada por um conjunto de camadas (que não sejam o classificador) conectadas e uma última camada com o classificador.

Os outros três tipos de camadas que compõem uma rede convolucional são: camada convolucional, de pooling e de retificação. A seguir será explicado cada uma delas. Para essas explicações, assuma que a entrada é uma imagem em escala de cinzas de 16x16 pixels. Assim, pode-se enxergar a entrada como uma matrix de 16 linhas com 16 colunas. Para simplificar ainda mais, sem perder a generalidade, assuma que cada célula apresenta o valor 1 se apresentar cor e 0 caso contrário.

# Camada Convolucional

A camada convolucional é a aplicação de um pequeno filtro, também chamado de kernel, em várias partes da imagem. Repare que o mesmo filtro é utilizado em todas as partes da imagem. Isso torna esse problema embarassosamente paralelizavel, o que justifica o uso de GPUs para fazer esse processamento. Cada kernel representa uma unidade de processamento, ou seja, um neurônio. A saída desse neurônio será ativada caso a parte da imagem case com o filtro. Na prática, a saída de cada kernel é uma submatriz formada pela multiplicação escalar de cada elemento do kernel pelo elemento correspondente na imagem de entrada.
A imagem abaixo essa ideia usando uma parade representando a entrada e diversas lanternas representando o kerkel.

![Exemplo de filtro convolucional em uma parede observando pela frente.](figs/conv_wall_front.png)
![Exemplo de filtro convolucional em uma parede observando por cima.](figs/conv_wall_top.png)

Quando se utiliza uma camada convolucional em uma rede, deve-se levar em consideração o problema do "zero-padding". Existem 3 casos a serem levados em consideração, sengundo a nomeclatura do MATLAB, considere um vetor como entrada de tamanho 'm' e um filtro (kernel) de tamanho 'k':
 - Válida (valid)
Neste tipo de convolução, todos os campos do filtro sempre devem estar associados a um elemento da entrada. Com isso, o tamanho do resultado gerado por essa camada será reduzido, ficando um vetor de tamanho 'm'-'k'+1. A figura abaixo demonstra esse comportamento.
[IMAGEM]
Isso impacta na definição da arquitetura da rede, pois limita a quantidadade de camadas convolucionais que podem ser utilizadas. Por outro lado, todos os pixels de entrada tem a mesma chance de incluênciar os pixels de saída.

 - Mesma (same)
Neste tipo de convolução, o tamanho da saída será o mesmo da saída. Para isso, é utilizado um artifício onde de 'padding' na entrada, assim, nas extremidades o filtro não é aplicado por completo, apenas em alguns pixels da entrada de tal forma, que seja possível criar um vetor de saído de mesmo tamanho que a entrada. A figura abaixo demonstra esse comportamento.
[IMAGEM]
Usando esse tipo de convolução, podemos ter quantas camadas convolucionais o hardware soportar. O problema dessa abordage é que as bordas da entrada são menos favorecidas, pois participam menos da construção do filtro do que os pixels internos. Lembre-se que o filtro aplicado em todas as posições são os mesmos. Dessa forma, utilizando esse tipo de convolução é mais difícil identificar características mapeadas nos extremos da entrada.

 - Completa (full)
Usando esse tipo de convolução, temos a mesma participaçao de todos os pixels de entrada. Isso porque é adicionado um 'padding' que permita isso. Dessa forma, o tamanho da entrada acaba crescendo, ficando com 'm'+'k'-1. A imagem abaixo demonstra esse comportamento.
[IMAGEM]
Usando esse tipo de convolução, todos os pixels tem a mesma oportunidade para definir o filtro, porém a saída vai se tornando muito grande. Além disso, a definição do filtro vai se tornando de difícil convergência devido a adição de novas bordas utilizando valores do 'padding'. Em geral, as convoluções mais utilizadas em redes neurais convolucionais (CNN) são: a 'válida' e a 'mesma'.

Para se conseguir esses tipos de convoluções, é necessário utilizar 'stride'. O 'stride' é o que define quantas posições o filtro será deslocado de uma aplicação para outra. Isso também causa um outro efeito chamado 'downsampling'.
O 'downsampling' é o descarte de parte da informação em detrimento de um processamento mais rápido. Isso ocorre quando temos uma redução no tamanho da entrada. A imagem abaixo ilustra esse efeito em dois passos: Processamento de uma convolução válida e posteriormente a execução de um 'downsampling' para reduzir o tamanho da saída (assumindo o prejuízo de perda de informação). Utilizando 'stride' não há necessidade de executar o processo intermediário. 
[IMAGEM]

# Camada de Pooling

# Camada de retificação
