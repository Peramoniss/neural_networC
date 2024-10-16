#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../include/bamboo.h"
#define debug 0
#define MAX_ATTEMPTS 3

// ######################################################################################################################

dataset *load_csv(FILE *file, char *delim, short header, int row_max_size, short force_not_null)
{

    char linha[row_max_size]; // Para armazenar uma linha do CSV
    char *token;               // Para armazenar cada célula da linha

    // Verifica se o arquivo foi aberto corretamente
    if (file == NULL)
    {
        printf("Erro ao abrir o arquivo!\n");
        return NULL;
    }

    int num_colunas = 0, num_linhas = 0;
    fgets(linha, sizeof(linha), file); // le o header/primeira linha
    token = strtok(linha, delim);
    while (token != NULL)
    {
        num_colunas++; // conta quantas colunas possui
        token = strtok(NULL, delim);
    }
    while (fgets(linha, sizeof(linha), file))
    {
        num_linhas++; // conta quantas linhas
    }

    if (!header)
        num_linhas++; // ignorou a primeira linha, pensando ser header. Portanto,
                      // precisa incrementar

    dataset *dados = (dataset *)malloc(sizeof(dataset));
    dados->columns = num_colunas;
    dados->rows = num_linhas;
    dados->matrix = (double **)malloc(num_linhas * sizeof(double *));
    for (int i = 0; i < num_linhas; i++)
    {
        dados->matrix[i] = (double *)calloc(num_colunas, sizeof(double));
    }

    rewind(file); // volta para o comeco
    if (header)
        fgets(linha, sizeof(linha), file); // ignora header

    // printf("Passou!\n");

    int i = 0, j = 0;
    // Lê o arquivo linha por linha
    while (fgets(linha, sizeof(linha), file))
    {

        // Remove o '\n' no final da linha, se houver
        linha[strcspn(linha, "\n")] = '\0';
        // printf("%s - %d %d\n", linha, num_linhas, num_colunas);

        // Divide a linha em colunas usando strtok
        token = strtok(linha, delim);
        while (token != NULL)
        {
            // printf("%s ", token);  // Exibe cada valor da coluna
            dados->matrix[i][j] = atof(token);
            token = strtok(NULL, delim); // Pega o próximo valor
            if (token == NULL &&
                j < num_colunas - 1)
            { // verifica se é um valor nulo e não no fim da linha
                printf(
                    "ERRO: Valor nulo não esperado encontrado na linha %d, campo %d\n",
                    i, j);
                if (force_not_null) // se o usuario pediu para encerrar o programa ao
                                    // encontrar valores nulos, encerra
                    exit(1);
            }
            j++;
        }

        // printf("\n");  // Quebra de linha após cada linha do CSV
        j = 0;
        i++;
    }
    // printf("Passou!\n");

    // Fecha o arquivo
    fclose(file);

    return dados;
}

void min_max_scale_column(dataset *data, int pos){
    double **matriz = data->matrix;
    double maior = matriz[0][pos];
        double menor = matriz[0][pos];

        for (int i = 1; i < data->rows; i++)
        {
            if (matriz[i][pos] > maior)
                maior = matriz[i][pos];
            else if (matriz[i][pos] < menor)
                menor = matriz[i][pos];
        }

        // uma vez encontrados os maiores e menores valores, aplica escalonamento
        // minmax
        for (int i = 0; i < data->rows; i++)
        {
            if (maior - menor != 0)
                matriz[i][pos] = (matriz[i][pos] - menor) / (maior - menor);
            else
                matriz[i][pos] = 0;
        }
}

void min_max_scale_data(dataset *data)
{
    double **matriz = data->matrix;

    // percorre uma coluna de cada vez, encontrando maior e menor valor
    for (int j = 0; j < data->columns; j++)
    {
        double maior = matriz[0][j];
        double menor = matriz[0][j];

        for (int i = 1; i < data->rows; i++)
        {
            if (matriz[i][j] > maior)
                maior = matriz[i][j];
            else if (matriz[i][j] < menor)
                menor = matriz[i][j];
        }

        // uma vez encontrados os maiores e menores valores, aplica escalonamento
        // minmax
        for (int i = 0; i < data->rows; i++)
        {
            if (maior - menor != 0)
                matriz[i][j] = (matriz[i][j] - menor) / (maior - menor);
            else
                matriz[i][j] = 0;
        }
    }
}

void x_y_split(dataset *original, dataset **x, dataset **y, int output_size)
{
    // Alocando memória para a matriz Y
    int rows = original->rows;
    *y = (dataset *)malloc(sizeof(dataset));
    (*y)->rows = rows;
    (*y)->columns = output_size;
    (*y)->matrix = (double **)malloc(rows * sizeof(double *));
    for (int i = 0; i < rows; i++)
    {
        (*y)->matrix[i] = (double *)malloc(output_size * sizeof(double));
    }

    // Alocando memória para a matriz X
    int remaining_cols = original->columns - output_size;
    *x = (dataset *)malloc(sizeof(dataset));
    (*x)->rows = rows;
    (*x)->columns = remaining_cols;
    (*x)->matrix = (double **)malloc(rows * sizeof(double *));
    for (int i = 0; i < rows; i++)
    {
        (*x)->matrix[i] = (double *)malloc(remaining_cols * sizeof(double));
    }

    // Copiando os dados para y e x
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < remaining_cols; j++)
        {
            (*x)->matrix[i][j] = original->matrix[i][j]; // Copia para X
        }
        for (int j = 0; j < output_size; j++)
        {
            (*y)->matrix[i][j] =
                original->matrix[i][j + remaining_cols]; // Copia para Y
        }
    }
}

void train_test_split(dataset *data, dataset **train, dataset **test, double train_ratio, int random_state, int output_size, enum split_method method)
{
    srand(random_state); // seta seed aleatoria

    *train = (dataset*)malloc(sizeof(dataset));
    *test = (dataset*)malloc(sizeof(dataset));
    // define separacao das instancias de treino e teste
    int instancias_totais = data->rows;
    int instancias_treino = instancias_totais * train_ratio;
    int instancias_teste = instancias_totais - instancias_treino;
    int *indices = (int *)malloc(instancias_totais * sizeof(int));

    if(method == PEREIRA){
        // define ordem de leitura das instancias do dataset original
        for (int i = 0; i < instancias_totais; i++)
        {
            int nova_instancia = rand() % instancias_totais;
            for (int j = 0; j < i; j++)
            {
                if (indices[j] == nova_instancia)
                {
                    nova_instancia++;
                    if (nova_instancia >= instancias_totais)
                        nova_instancia = 0;
                    j = 0;
                }
            }

            indices[i] = nova_instancia;
        }
    }else{
        //embaralhamento Fisher-Yates
        for (int i = 0; i < instancias_totais; i++) {
            indices[i] = i; // Preenche com 0, 1, 2, ..., instancias_totais - 1
        }

        // Embaralha os índices usando Fisher-Yates
        for (int i = instancias_totais - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
    }

    // define proporcao das classes
    int *class_ratio, *other_class_ratio, *class_counter;
    int total = 0;
    class_ratio = (int *)calloc(output_size, sizeof(int));
    class_counter = (int *)calloc(output_size, sizeof(int));
    other_class_ratio = (int *)calloc(output_size, sizeof(int));

   for (int i = 0; i < instancias_totais; i++)
    {
        double max_value = data->matrix[i][data->columns - output_size];  // Variável para armazenar o maior valor
        int max_class = 0;     // Variável para armazenar o índice da classe com maior valor

        // Percorre as saídas da instância e encontra a maior classe
        for (int j = 1; j < output_size; j++)
        {
            double current_value = data->matrix[i][data->columns - output_size + j];
            // Atualiza se o valor atual é maior que o valor máximo encontrado até agora
            if (current_value > max_value)
            {
                max_value = current_value;
                max_class = j; // Armazena a classe correspondente ao maior valor
            }
        }

        class_counter[max_class]+=1.0; // Incrementa a quantidade de instâncias para essa classe
        total+=1;  // Incrementa o total de instâncias
    }

    printf("Total: %d   -   ", total);
    for (int i = 0; i < output_size; i++)
        printf("* %d   ", class_counter[i]);


    int remaining_instances = instancias_treino;
    for (int i = 0; i < output_size; i++)
    {
        printf("%f", round(instancias_treino * (class_counter[i]*1.0 / total)));
        class_ratio[i] = (int)round(instancias_treino * (class_counter[i]*1.0 / total)); // define quantas instancias de cada classe devem estar no conjunto de treino
        remaining_instances -= class_ratio[i];
    }

    while(remaining_instances > 0){
        class_ratio[remaining_instances % output_size]++;
        remaining_instances--;
    }

    printf("Total: %d   -   \n", instancias_treino);
    for (int i = 0; i < output_size; i++)
    {
        printf("Classe %d: %d   -   \n", i, class_ratio[i]);
    }

    (*train)->rows = instancias_treino;
    (*train)->columns = data->columns;
    (*train)->matrix = (double **)malloc(instancias_treino * sizeof(double *));
    for (int i = 0; i < instancias_treino; i++)
        (*train)->matrix[i] = (double *)malloc((data->columns) * sizeof(double));

    (*test)->rows = instancias_teste;
    (*test)->columns = data->columns;
    (*test)->matrix = (double **)malloc(instancias_teste * sizeof(double *));
    for (int i = 0; i < instancias_teste; i++)
        (*test)->matrix[i] = (double *)malloc(data->columns * sizeof(double));

    int **instancias_alocadas = (int **)malloc(2 * sizeof(int*));
    for (int p = 0; p < output_size; p++)
        instancias_alocadas[p] = (int *)calloc(output_size, sizeof(int));

    int i, train_counter = 0, test_counter = 0,  loop_detector = 0;
    int last_indices[MAX_ATTEMPTS] = {-1}; // Armazena os últimos índices processados
    int attempt_count = 0; // Contador de tentativas

    for (i = 0; i < output_size; i++){
        other_class_ratio[i] = class_counter[i] - class_ratio[i];
    }

    for (i = 0; i < total; i++)
    {
        int current_instance_class = 0;
        // Inicializa current_biggest_value com um valor baixo
        int current_biggest_value = data->matrix[indices[i]][data->columns - output_size];  // Assumindo que os valores são não negativos
        
        //printf("Index: %d\n", indices[i]); // Verificando qual índice está sendo usado
        //printf("Valor inicial: %f\n", data->matrix[indices[i]][data->columns - output_neurons]);

        // procura classe da instancia atual, baseado nos indices embaralhados
        for (int t = 1; t < output_size; t++) // t = 0 pois a classe 0 pode ser a maior também
        {
            // Verifica os valores antes da comparação
            float current_value = data->matrix[indices[i]][data->columns - output_size + t];
            //printf("Verificando classe %d: valor = %f\n", t, current_value);

            // Atualiza current_biggest_value
            if (current_value > current_biggest_value)
            {
                current_biggest_value = current_value;
                current_instance_class = t;
            }
        }

        //printf("Index: %d, Current Class: %d, Current Biggest Value: %f\n", indices[i], current_instance_class, current_biggest_value);

        for (int j = 0; j < MAX_ATTEMPTS; j++)
        {
            if (last_indices[j] == indices[i]) {
                loop_detector = 1; // Encontrou um índice repetido
                break;
            }
        }
        last_indices[attempt_count % MAX_ATTEMPTS] = indices[i];
        attempt_count++;
        if (attempt_count < MAX_ATTEMPTS) loop_detector = 0;



        // se o limite de instancias dessa classe ja foi alocado
        // printf("Conferindo se %d < %d = %d OU se %d < %d. Loop? %d\n", instancias_alocadas[0][current_instance_class], class_ratio[current_instance_class], 
        // instancias_alocadas[0][current_instance_class] < class_ratio[current_instance_class], instancias_alocadas[1][current_instance_class], other_class_ratio[current_instance_class], loop_detector);
        if (instancias_alocadas[0][current_instance_class] < class_ratio[current_instance_class])
        {    
            //printf("Entrou em treino em indice %d   -   ", train_counter);
            for (int j = 0; j < data->columns; j++)
            {
                (*train)->matrix[train_counter][j] = data->matrix[indices[i]][j]; // treino recebe instancia do original
            }
            train_counter++;
            instancias_alocadas[0][current_instance_class]++; // incrementa instancias daquela classe que foram alocadas
        }else if(instancias_alocadas[1][current_instance_class] < other_class_ratio[current_instance_class]){
            //printf("Entrou em teste em indice %d   -   ", test_counter);
            for (int j = 0; j < data->columns; j++)
            {
                (*test)->matrix[test_counter][j] = data->matrix[indices[i]][j]; // treino recebe instancia do original
            }
            test_counter++;
            instancias_alocadas[1][current_instance_class]++; // incrementa instancias daquela classe que foram alocadas
        }else{
            printf("ERROR");
            return;
        }
    }
}

void to_csv(dataset *data, char *dir){
    FILE *exported;
    if(dir != NULL){
        exported = fopen(dir, "wt");
    }else{
        exported = fopen("export.csv", "wt");
    }

    for(int i = 0; i < data->rows; i++){
        for(int j = 0; j < data->columns; j++){
            fprintf(exported, "%f,", data->matrix[i][j]);
        }
        fprintf(exported, "\n");
    }

    fclose(exported);
}

void dataframe_head(dataset *data, int head)
{
    if (head > data->rows)
    {
        printf("Não ha instancias suficientes\n");
        return;
    }
    for (int i = 0; i < head; i++)
    {
        for (int j = 0; j < data->columns; j++)
        {
            printf("%.4f ", data->matrix[i][j]);
        }
        printf("\n");
    }
}
