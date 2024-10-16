#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../include/bamboo.h"
#include "../include/MLP.h"

int main()
{
    FILE *arquivo = fopen("export.txt", "r");
    neural_network *nn = load_neural_network(arquivo);
    FILE *data = fopen("examples/nba_dados_sem_games.csv", "r");
    dataset *dados = load_csv(data, ",", 1, 1000, 1);
    min_max_scale_column(dados, 1); //AGE - é uma quantidade, nao pode continuar categorica
    get_dummies(dados);
    printf("%d colunas pos-dummy\n", dados->columns);
    min_max_scale_data(dados);

    dataset *x, *y;
    x_y_split(dados, &x, &y, nn->output_neurons);
    
    get_results_report(nn, x, y, 1);

    print_confusion_matrix(nn->report);
    
    printf("Amem\n");
    return 0;
}