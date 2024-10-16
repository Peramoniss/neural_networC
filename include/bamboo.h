#ifndef BAMBOO_H
#define BAMBOO_H

#include <stdio.h>


typedef struct dataset
{
    double **matrix; // matriz que conecta os neuronios pelos seus pesos
    int columns;     // mantem forma da matriz - linhas é o número de neuronios de
                     // entrada e colunas os de saida
    int rows;
} dataset;

// Enums
// ######################################################################################################################

enum split_method{
    PEREIRA,
    FISCHER_YATES
};


dataset *load_csv(FILE *file, char *delim, short header, int row_max_size, short force_not_null);

void min_max_scale_column(dataset *data, int pos);

void min_max_scale_data(dataset *data);

void x_y_split(dataset *original, dataset **x, dataset **y, int output_size);

void train_test_split(dataset *data, dataset **train, dataset **test, double train_ratio, int random_state, int output_size, enum split_method method);

void to_csv(dataset *data, char *dir);

void dataframe_head(dataset *data, int head);

#endif // BAMBOO_H
