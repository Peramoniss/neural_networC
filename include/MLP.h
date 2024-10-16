#ifndef MLP_h
#define MLP_h

#include "bamboo.h"

#define fit train
#define fit_with_early_stopping train_with_early_stopping

// Enums
// ######################################################################################################################

enum activation_function {
    SIGMOID,
    RELU,
    TANH
};

// ######################################################################################################################

// Structs
// ######################################################################################################################
typedef struct weight_matrix
{
    double **matrix; // matriz que conecta os neuronios pelos seus pesos
    int columns;     // mantem forma da matriz - linhas é o número de neuronios de entrada e colunas os de saida
    int rows;
} weight_matrix;

//futuramente, adaptar para usar dicionario
typedef struct metrics
{
    double global_accuracy;
    int classes;
    int **confusion_matrix;
    double **class_metrics; //rows equivalent to classes, 3 columns - precision, recall, f-1. 
} metrics;

//unused yet 
typedef struct dict{
    char *key;
    void *value;
    char *type; //define o tipo para castar o valor
} dictionary;

typedef struct layer
{
    double *values; // vetor com dados dos neuronios e uma camada
    int neurons; // indica quantidade de neuronios na camada
} layer;          // chamo de neuronios?

typedef struct neural_network
{
    layer *layers;
    layer *errors;
    weight_matrix *weights;
    metrics *report;
    int input_neurons;
    int output_neurons;
    int hidden_layers;
    int total_layers;
    enum activation_function activation;
} neural_network;

// ######################################################################################################################

// Functions
// ######################################################################################################################


void get_dummies(dataset *data);

int dummy(dataset *data, int column);

neural_network *create_neural_network(dataset *data, int output_neurons, int hidden_layers, int *hidden_layers_size, enum activation_function activation_function, int random_state);

void train(neural_network *network, dataset *x_data, dataset *y_data, double learning_rate, double momentum, int epochs, int monitor);

void train_with_early_stopping(neural_network *network, dataset *x_data, dataset *y_data, double learning_rate, double momentum, int epochs, int monitor, double min_loss, int patience);

void print_confusion_matrix(metrics *report);

int *predict(double **x, double **y, int instances, neural_network *network, int log);

void get_results_report(neural_network *network, dataset *x, dataset *y, short log);

void export_neural_network(neural_network *nn, char *dir);

neural_network * load_neural_network(FILE *source);

#endif