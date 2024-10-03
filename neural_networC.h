#ifndef NEURAL_NETWORC_INCLUDED
#define NEURAL_NETWORC_INCLUDED
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>


//Structs
//######################################################################################################################
typedef struct pesos {
    float **matriz; //matriz que conecta os neuronios pelos seus pesos
    int colunas; //mantem forma da matriz - linhas é o número de neuronios de entrada e colunas os de saida
    int linhas;
} pesos;

typedef struct dataset{
    float **matriz; //matriz que conecta os neuronios pelos seus pesos
    int colunas; //mantem forma da matriz - linhas é o número de neuronios de entrada e colunas os de saida
    int linhas;
} dataset;

typedef struct camada {
    float *valor; //vetor com dados dos neuronios e uma camada 
    int neuronios; //indica quantidade de neuronios na camada
} camada; //chamo de neuronios?

typedef struct neural_network
{
    dataset *dados;
    camada *camadas;
    camada *erros;
    pesos *rede;
    pesos *delta;
    int output_neurons;
    int hidden_layers;
} neural_network;

//######################################################################################################################

//Neural_Network
//######################################################################################################################
float** inicia_matriz(int neuronios, int saida);

int desaloca_matriz(int **matriz, int neuronios);

pesos *gera_rede_neural(int n_camadas_intermediarias, int neuronios_entrada, int neuronios_saida, camada **camadas);

float funcao_sigmoide(float x);

int calculate(pesos **peso, camada **camadas, int n_camadas);

int carrega_camada_entrada(float *dados, float *camada, int tamanho);

neural_network * cria_rede_neural(dataset *dados, int neuronios_saida, int camadas_intermediarias);

int backpropagation(camada **camadas, camada **erro, pesos **peso, pesos **previous_delta, float *saida_esperada, int num_camadas, float momentum, float alpha );

int error_factor(camada **camadas, camada **camada_erros, pesos **rede, int num_camadas, float *saida_esperada);

int treina_rede_neural(neural_network *n, dataset *saidas_esperadas, int epocas, float momentum, float alpha);
//######################################################################################################################

//Data treatment
//######################################################################################################################
dataset *carrega_dados_csv(FILE* arquivo, char *delim, int tam_max_linha, short force_not_null);

int escalona_dados(dataset *dados);

int dataframe_head(dataset *ata, int head);

int x_y_split(dataset *original, dataset **x, dataset **y, int output_neurons);
//######################################################################################################################

#endif // NEURAL_NETWORC_INCLUDED