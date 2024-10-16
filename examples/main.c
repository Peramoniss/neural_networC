#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../include/bamboo.h"
#include "../include/MLP.h"

int main()
{
    FILE *arquivo = fopen("examples/nba_dados_sem_games.csv", "r");
    dataset *dados = load_csv(arquivo, ",", 1, 1000, 1);
    min_max_scale_column(dados, 1); //AGE - é uma quantidade, nao pode continuar categorica
    //min_max_scale_column(dados, 0); //GAMES - é uma quantidade, nao pode continuar categorica
    get_dummies(dados);
    printf("%d colunas pos-dummy\n", dados->columns);

    int neuronios_saida = 2, camadas_ocultas = 1;
    dataset *test_data_splitted, *train_data_splitted;
    train_test_split(dados, &train_data_splitted, &test_data_splitted, 0.7, 8, neuronios_saida, PEREIRA);
    min_max_scale_data(train_data_splitted);

    int vetor_entradas[] = {20,10};
    dataset *x, *y;
    x_y_split(train_data_splitted, &x, &y, neuronios_saida);
    neural_network *nn = create_neural_network(train_data_splitted, neuronios_saida, camadas_ocultas, vetor_entradas, SIGMOID, time(NULL)); //pode ser null

    fit_with_early_stopping(nn, x, y, 0.2, 1, 1000, 10, 0.0001, 1);
    //train(nn, x, y, 0.1, 1, 1000, 10);

    min_max_scale_data(test_data_splitted);

    printf("Total data: %d\n", dados->rows);
    printf("Train data: %d\n", train_data_splitted->rows);
    printf("Test data: %d\n", test_data_splitted->rows);
    dataset *test_x, *test_y;
    x_y_split(test_data_splitted, &test_x, &test_y, neuronios_saida);

    get_results_report(nn, test_x, test_y, 1);

    export_neural_network(nn, NULL);
    to_csv(train_data_splitted, "trained_data.csv");
    to_csv(test_data_splitted, "tested_data.csv");

    print_confusion_matrix(nn->report);
    
    printf("Amem\n");
    return 0;
}