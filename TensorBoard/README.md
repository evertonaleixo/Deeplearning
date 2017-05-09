# TensorBoard

O TensorBoard é uma ferramenta que vem instalada junto ao pacote do Tensorflow quando instalado com o pip.

```
pip install tensorflow
```

O projeto possui uma página [no github](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tensorboard).

O TensorBoard é uma aplicação web utilizada para compreender os modelos desenvolvidos em tensorflow. Além disso, ele apresenta excelentes modos de visualização dos dados.

## Executar o TensorBoard
```
tensorboard --logdir=path/to/logs
```

O 'path/to/logs' é o diretório onde o tensorflow armazena as informações necessárias para a apresentação dos dados na aplicação web. Para a aplicação em tensorflow gerar esses dados, é necessário a adição de algumas linhas de código. Em geral, é necessário adicionar valores no 'tf.summary.histogram', 'tf.summary.image', 'tf.summary.scalar' e o escritor 'tf.summary.FileWriter(path/to/file)'.

O [exemplo](./mnist/) apresenta um código feito em python usando tensorflow com anotações afim de gerar dados para o TensorBoard.


## Possíveis problemas
Algumas bibliotecas podem estar desatualizadas. Execute os seguintes comandos em um terminal:
```
pip install --upgrade psutil
pip install --upgrade tensorflow
```
