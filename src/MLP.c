#include "../include/bamboo.h"
#include "../include/MLP.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#define debug 0

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double relu(double x) {
    return (x > 0) ? x : 0;
}

double tanh_activation(double x) {
    return tanh(x);
}

double sigmoid_derivative(double x)
{
    return x * (1.0 - x);
}

double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}

double tanh_derivative(double x) {
    return 1 - (tanh(x) * tanh(x));
}


// Função para alocar memória para matrizes
double **allocate_matrix(int rows, int cols)
{
    double **matrix = (double **)calloc(rows, sizeof(double *));
    for (int i = 0; i < rows; i++)
    {
        matrix[i] = (double *)calloc(cols, sizeof(double));
    }

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            matrix[i][j] = ((double)rand() / RAND_MAX) * 2 - 1; // Valores entre -1 e 1
        }
    }

    return matrix;
}

neural_network *create_neural_network(dataset *data, int output_neurons, int hidden_layers, int *hidden_layers_size, enum activation_function activation_function, int random_state) {
    neural_network *network = (neural_network *)malloc(sizeof(neural_network));

    network->activation = activation_function;
    network->hidden_layers = hidden_layers;
    network->total_layers = hidden_layers + 2;
    network->output_neurons = output_neurons;

    // cria camadas da rede
    layer *camadas = (layer *)malloc(network->total_layers * sizeof(layer));
    int input_neurons = data->columns;

    camadas[0].values = (double *)calloc(input_neurons, sizeof(double)); // inicia vetor da camada de entrada
    camadas[0].neurons = input_neurons;
    camadas[network->total_layers - 1].values = (double *)calloc(output_neurons, sizeof(double)); // inicia vetor da camada de saida
    camadas[network->total_layers - 1].neurons = output_neurons;


    if(hidden_layers_size == NULL){
        int occult_neurons = (input_neurons + output_neurons) / 2; // define numero de neuronios na camada intermediária (pensar em deixar o usuario configurar)
        // int occult_neurons = 3;
        for (int i = 1; i < network->total_layers - 1; i++)
        {                                                                        // para cada camada intermediária
            camadas[i].values = (double *)calloc(occult_neurons, sizeof(double)); // inicia vetor da camada
            camadas[i].neurons = occult_neurons;                               // armazena tamanho da camada
            occult_neurons = (occult_neurons + output_neurons) / 2;              // atualiza o tamanho da próxima camada
            // int occult_neurons = 3;
        }
    }else{
        for(int i = 0; i < network->hidden_layers; i++){
            printf("Size %d: %d neuronois\n", sizeof(hidden_layers_size), hidden_layers_size[i]);
            camadas[i+1].values = (double *)calloc(hidden_layers_size[i], sizeof(double));
            camadas[i+1].neurons = hidden_layers_size[i];
        }
    }

    network->layers = camadas; // passa variavel criada para estrutura da rede neural
    network->input_neurons = input_neurons;

    // cria matrizes de pesos
    weight_matrix *weights = (weight_matrix *)malloc((network->total_layers - 1) * sizeof(weight_matrix)); // Ex. se tem 3 camadas, haverá 2 matrizes de peso
    srand(random_state);

    for (int i = 0; i < network->total_layers - 1; i++)
    {
        weights[i].matrix = allocate_matrix(network->layers[i].neurons, network->layers[i + 1].neurons); // já recebe matriz com valores aleatórios
        weights[i].rows = network->layers[i].neurons;                                                     // cada matriz de pesos usa linhas para a camada atual e colunas para a seguinte
        weights[i].columns = network->layers[i + 1].neurons;
    }

    network->weights = weights; // passa variavel criada para estrutura da rede neural
    network->report = NULL;

    return network;
}

void feedforward(neural_network *network)
{
    for (int k = 1; k < network->total_layers; k++) { // para cada camada, fora a de entrada
        for (int i = 0; i < network->layers[k].neurons; i++)
        { // para cada neuronio da camada atual
            for (int j = 0; j < network->layers[k - 1].neurons; j++)
            { // para cada neuronio da camada anterior
                // soma ao valor do neuronio a multiplicacao entre cada neuronio da camada anterior e o peso entre o neuronio na camada anterior e o da camada atual
                // matriz e na [j][i] pois j representa a camada anterior, e na alocacao, linhas sao a atual e colunas a seguinte
                // weights e na k-1 pois possui uma posicao a menos que as camadas, e esta representando a camada anterior com a atual
                network->layers[k].values[i] += network->layers[k - 1].values[j] * network->weights[k - 1].matrix[j][i];
            }
            
            switch (network->activation)
            {
            case SIGMOID:
                network->layers[k].values[i] = sigmoid(network->layers[k].values[i]);
                break;
            case RELU:
                network->layers[k].values[i] = relu(network->layers[k].values[i]);
                break;
            case TANH:
                network->layers[k].values[i] = tanh_activation(network->layers[k].values[i]);
                break;
            default:
                network->layers[k].values[i] = sigmoid(network->layers[k].values[i]);
                break;
            }
        }
    }
}

void backpropagation(neural_network *network, double *target, double learning_rate, double momentum) {
    network->errors = (layer *)malloc((network->total_layers - 1) * sizeof(layer)); // camada de entrada nao possui pesos
    for (int i = 0; i < network->total_layers - 1; i++)
    { // para cada camada tirando a de entrada
        // i+1 pois ignora entrada
        network->errors[i].values = calloc(network->layers[i + 1].neurons, sizeof(double));
        network->errors[i].neurons = network->layers[i + 1].neurons;
    }
    // printf("Criou erros. Output neurons = %d\n", network->output_neurons);

    // Erro na camada de saída
    double *output_error = (double *)malloc(network->output_neurons * sizeof(double));
    double *output_delta = (double *)malloc(network->output_neurons * sizeof(double));

    for (int i = 0; i < network->output_neurons; i++)
    {
        double output_received = network->layers[network->total_layers - 1].values[i];
        output_error[i] = target[i] - output_received;                           // calcula diferenca entre saida esperada e recebida
        output_delta[i] = output_error[i] * sigmoid_derivative(output_received); // calcula erro
        
    }

    network->errors[network->total_layers - 2].values = output_delta; // vetor de erros na ultima posicao recebe os erros do output (Ex. 3 camadas. Erros tera 2 camadas. 3-2 = posicao 1, exata posicao onde os erros ficarao)

    // Erro nas camadas ocultas
    for (int k = network->total_layers - 2; k >= 1; k--)
    { // para cada camada oculta (Ex. 2 camadas ocultas - para k = 1 ate k = 2, duas vezes)
        double *hidden_error = (double *)malloc(network->layers[k].neurons * sizeof(double));
        double *hidden_delta = (double *)malloc(network->layers[k].neurons * sizeof(double));

        for (int i = 0; i < network->layers[k].neurons; i++)
        {                        // para cada neuronio na camada oculta
            hidden_error[i] = 0; // erro comeca em zero
            for (int j = 0; j < network->layers[k + 1].neurons; j++)
            { // e se torna somatoria dos erros da camada posterior ponderados pelos pesos
                /*multiplica o erro da camada posterior (fica no indice k pois ha uma camada de erros a menos) pelo peso da conexao
                (fica em k pois possui uma camada a menos, sendo i a linha - camada anterior - e j a coluna - camada posterior)*/
                hidden_error[i] += network->errors[k].values[j] * network->weights[k].matrix[i][j];
            }
            double output_received = network->layers[k].values[i];
            if (network->activation == SIGMOID) {
                hidden_delta[i] = hidden_error[i] * sigmoid_derivative(output_received);
            }
            else if (network->activation == RELU) {
                hidden_delta[i] = hidden_error[i] * relu_derivative(output_received);
            }
            else if (network->activation == TANH) {
                hidden_delta[i] = hidden_error[i] * tanh_derivative(output_received);
            }
        }

        network->errors[k - 1].values = hidden_delta; // vetor de erros na posicao anterior recebe os erros do output (Ex. 3 camadas. Erros tera 2 camadas. k=1, posicao 0 recebera o erro, pois representa a camada oculta. Posicao 1 representa a de saida)
        free(hidden_error);
        free(hidden_delta);
    }

    // backpropagation
    for (int k = 0; k < network->total_layers - 1; k++)
    { // vetor de pesos possui uma posicao a menos
        for (int i = 0; i < network->layers[k].neurons; i++)
        {
            for (int j = 0; j < network->layers[k + 1].neurons; j++)
            {
                /*matriz de pesos entre camada atual e posterior recebe multiplicacao entre alfa (learning rate) e o valor de saida
                do feedforward da camada atual multiplicado pelo erro do neuronio da camada de saida (k+1, representado por k pois erros
                nao possuem camada de entrada)*/
                network->weights[k].matrix[i][j] = network->weights[k].matrix[i][j] * momentum + learning_rate * network->layers[k].values[i] * network->errors[k].values[j];

            }
        }
    }

    free(output_error);
    free(output_delta);
}

void load_input_layer(double *dados, double *camada, int tamanho)
{
    for (int i = 0; i < tamanho; i++)
    {
        camada[i] = dados[i];
    }
}

// double **inputs, double **targets, int num_samples, double learning_rate, int epochs
void train(neural_network *network, dataset *x_data, dataset *y_data, double learning_rate, double momentum, int epochs, int monitor) {
    for (int epoch = 0; epoch < epochs; epoch++)
    { // para cada época
        for (int i = 0; i < x_data->rows; i++)
        { // para cada instancia do dataset
            load_input_layer(x_data->matrix[i], network->layers[0].values, network->layers[0].neurons);
            feedforward(network);
            backpropagation(network, y_data->matrix[i], learning_rate, momentum);
        }
        // Monitoramento do erro durante periodo estipulado pelo usuario
        if (monitor > 0 && epoch % monitor == 0)
        {
            double total_loss = 0;
            for (int i = 0; i < x_data->rows; i++)
            {
                load_input_layer(x_data->matrix[i], network->layers[0].values, network->layers[0].neurons);
                feedforward(network);
                for (int j = 0; j < network->output_neurons; j++)
                {
                    double error = y_data->matrix[i][j] - network->layers[network->total_layers - 1].values[j];
                    total_loss += error * error;
                }
            }
            printf("Epoch %d, Loss: %f\n", epoch, total_loss / x_data->rows);
        }
    }
}

void print_confusion_matrix(metrics *report){
    int maior = report->confusion_matrix[0][0];
    for(int i = 0; i < report->classes; i++){
        for(int j = 0; j < report->classes; j++){
            if(report->confusion_matrix[i][j] > maior)
                maior = report->confusion_matrix[i][j];
        }
    }

    int digits = 0;
    while(maior > 0){
        maior = maior / 10;
        digits++;
    }

    char formato[20]; // Um buffer para armazenar a string de formato

    // Cria dinamicamente o formato, por exemplo "%-10d"

    // Usa o formato criado dinamicamente no printf
    printf("MAtriz de confusao (linha = classe esperada; coluna = classe predita):\n");

    for(int i = 0; i < digits; i++)
        printf(" ");
        
    sprintf(formato, "%%-%dd |", digits);
    for(int i = 0; i < report->classes; i++){
        printf(formato, i);}

    printf("\n-----------------------------\n");
    for(int i = 0; i < report->classes; i++){
        printf("%d |", i);
        for(int j = 0; j < report->classes; j++){
            printf(formato, report->confusion_matrix[i][j]);
        }
        printf("\n");
    }
}

int *predict(double **x, double **y, int instances, neural_network *network, int log) {
    int *predicoes = (int*) malloc(instances * sizeof(int));

    for(int i = 0; i < instances; i++){
        load_input_layer(x[i], network->layers[0].values, network->input_neurons);
        feedforward(network);

        double predicao = network->layers[network->total_layers - 1].values[0];
        int predicao_classe = 0;
        //printf("Assume classe: %d; E valor: %f\n", predicao_classe, network->layers[network->total_layers - 1].values[0]);
        for (int j = 1; j < network->output_neurons; j++)
        {
            if (predicao < network->layers[network->total_layers - 1].values[j])
            {
                //printf("Nova classe: %d; Novo valor: %f; Valor anterior: %f\n", j, network->layers[network->total_layers - 1].values[j], predicao);
                predicao_classe = j;
                predicao = network->layers[network->total_layers - 1].values[j];
            }
        }

        double valor_classe_esperada = y[i][0];
        int classe_esperada = 0;
        for (int j = 1; j < network->output_neurons; j++)
        {
            if (y[i][j] > valor_classe_esperada)
            {
                valor_classe_esperada = y[i][j];
                classe_esperada = j;
            }
        }

        if (log) printf("Classe prevista: %d. Classe esperada %d", predicao_classe, classe_esperada);
        predicoes[i] = predicao_classe;
    }
    
    return predicoes;
} 

void get_results_report(neural_network *network, dataset *x, dataset *y, short log) {
    int *instancias_por_classe = (int *)calloc(network->output_neurons, sizeof(int));       // cria vetor do tamanho das classes para armazenar o total de instancias em cada
    int **previsoes_por_classe = (int **)malloc((network->output_neurons) * sizeof(int *)); // cria vetor do tamanho das classes para armazenar o total de previsoes em cada
    for (int i = 0; i < network->output_neurons; i++)
        previsoes_por_classe[i] = (int *)calloc(network->output_neurons, sizeof(int)); // cria vetor do tamanho das classes para armazenar o total de previsoes em cada
    int acertos = 0, total = x->rows;
    for (int i = 0; i < x->rows; i++)
    {
        load_input_layer(x->matrix[i], network->layers[0].values, network->layers[0].neurons);
        feedforward(network);

        double predicao = network->layers[network->total_layers - 1].values[0];
        int predicao_classe = 0;
        //printf("Assume classe: %d; E valor: %f\n", predicao_classe, network->layers[network->total_layers - 1].values[0]);
        for (int j = 1; j < network->output_neurons; j++)
        {
            if (predicao < network->layers[network->total_layers - 1].values[j])
            {
                //printf("Nova classe: %d; Novo valor: %f; Valor anterior: %f\n", j, network->layers[network->total_layers - 1].values[j], predicao);
                predicao_classe = j;
                predicao = network->layers[network->total_layers - 1].values[j];
            }
        }

        double valor_classe_esperada = y->matrix[i][0];
        int classe_esperada = 0;
        for (int j = 1; j < network->output_neurons; j++)
        {
            if (y->matrix[i][j] > valor_classe_esperada)
            {
                valor_classe_esperada = y->matrix[i][j];
                classe_esperada = j;
            }
        }

        previsoes_por_classe[classe_esperada][predicao_classe] += 1;
        instancias_por_classe[classe_esperada] += 1;

        if (predicao_classe == classe_esperada)
        {
            acertos++;
        }
    }


    double acuracia_geral = (double)acertos / total;

    //gera estrutura para armazenar os resultados
    metrics *result = (metrics *)malloc(sizeof(metrics));
    result->global_accuracy = acuracia_geral;
    result->class_metrics = malloc(network->output_neurons * sizeof(double *)); // cria vetor para cada classe
    for (int i = 0; i < network->output_neurons; i++) {
        result->class_metrics[i] = (double *)malloc(3 * sizeof(double)); // coloca uma posicao para cada metrica
    }

    if (log) printf("Resultados obtidos:\n");

    //verifica as metricas para cada classe
    for (int i = 0; i < network->output_neurons; i++) {
        if (log) printf("+-------------------------------------------------------\n");
        if (log) printf("                    Classe %d                    \n", i);
        if(log) printf("| Previsoes esperadas para classe %d: %d\n", i, instancias_por_classe[i]);
        int previsoes_incorretas = 0;
        for (int j = 0; j < network->output_neurons; j++) {
            if(log) printf("| Previsoes para %d: %d\n", j, previsoes_por_classe[i][j]);
            if (i != j)                                             // se nao for a classe correta
                previsoes_incorretas += previsoes_por_classe[j][i]; // soma nas classes incorretas
        }

        // previsoes_por_classe[i][i] = espera i, preve i = acertos. Acertos / acertos esperados
        double precisao;
        int previsoes_para_i = 0;
        for (int j = 0; j < network->output_neurons; j++)
            previsoes_para_i += previsoes_por_classe[j][i];

        if(previsoes_para_i > 0)
            precisao = (double)previsoes_por_classe[i][i] / previsoes_para_i;
        else
            precisao = 0;
        if(log) printf("| Precisao: %f\n", precisao * 100.0);

        // Acertos / total de previses para a classe
        double recall;
        if (instancias_por_classe[i] > 0)
            recall = (double)previsoes_por_classe[i][i] / instancias_por_classe[i];
        else
            recall = 0;
        if(log) printf("| Recall: %f\n", recall * 100.0);

        double f1;
        if( precisao != 0 || recall != 0)
            f1 = (2 * precisao * recall) / (precisao + recall);
        else
            f1 = 0;
        if(log) printf("| F-1 Score: %f\n", f1 * 100.0);

        result->class_metrics[i][0] = precisao;
        result->class_metrics[i][1] = recall;
        result->class_metrics[i][2] = f1;
        if (log) printf("+-------------------------------------------------------\n\n");

    }

    if(log) printf("| Acuracia: %d / %d = %f\n", acertos, total, acuracia_geral * 100.0);
    
    result->confusion_matrix = previsoes_por_classe;
    result->classes = network->output_neurons;

    network->report = result;

    free(instancias_por_classe);
}

void train_with_early_stopping(neural_network *network, dataset *x_data, dataset *y_data, double learning_rate, double momentum, int epochs, int monitor, double min_loss, int patience) {
    double last_loss = INFINITY;
    int patience_counter = 0;

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        for (int i = 0; i < x_data->rows; i++)
        {
            // Alimenta os dados e retropropaga o erro
            load_input_layer(x_data->matrix[i], network->layers[0].values, network->layers[0].neurons);
            feedforward(network);
            backpropagation(network, y_data->matrix[i], learning_rate, momentum);
        }

        // Monitoramento do erro (loss)
        if (monitor > 0 && epoch % monitor == 0)
        {
            double total_loss = 0;
            for (int i = 0; i < x_data->rows; i++)
            {
                load_input_layer(x_data->matrix[i], network->layers[0].values, network->layers[0].neurons);
                feedforward(network);
                for (int j = 0; j < network->output_neurons; j++)
                {
                    double error = y_data->matrix[i][j] - network->layers[network->total_layers - 1].values[j];
                    total_loss += error * error;
                }
            }
            total_loss /= x_data->rows;

            printf("Epoch %d, Loss: %f\n", epoch, total_loss);

            // Verificar se o loss melhorou
            if (last_loss - total_loss  > min_loss) {
                patience_counter = 0; // Resetar paciência
            }
            else {
                patience_counter++;
            }

            // Verificar se o loss nao esta melhoando abaixo da expectativa mínima desejada
            if (last_loss - total_loss < min_loss && patience_counter > patience)
            {
                printf("Treinamento interrompido: diferenca minimo de loss atingida em %f (diferenca: %f)\n", total_loss, last_loss - total_loss);
                break;
            }

            last_loss = total_loss;
        }
    }
}

neural_network * load_neural_network(FILE *source) {

    if (source == NULL) {
        printf("Error: file pointer is NULL\n");
        return NULL;
    }
    neural_network *network = (neural_network *) malloc(sizeof(neural_network));
    char linha[10000];

    fgets(linha, 1000, source);

    network->total_layers = atoi(strtok(linha, ","));
    network->hidden_layers = atoi(strtok(NULL, ","));

    fgets(linha, 1000, source);

    network->input_neurons = atoi(strtok(linha, ","));
    network->output_neurons = atoi(strtok(NULL, ","));
    

    fgets(linha, 1000, source);

    network->layers = (layer *)malloc(network->total_layers * sizeof(layer));
    network->layers[0].neurons = atoi(strtok(linha, ","));
    network->layers[0].values = (double *) malloc(network->layers[0].neurons * sizeof(double));

    for (int i = 1; i < network->total_layers; i++){
        network->layers[i].neurons = atoi(strtok(NULL, ","));
        network->layers[i].values = (double *) malloc(network->layers[i].neurons * sizeof(double));
    }

    network->weights = (weight_matrix *) malloc( (network->total_layers-1) * sizeof(weight_matrix));
    for(int k=0; k < network->total_layers-1; k++){
        network->weights[k].matrix = (double **) malloc(network->layers[k].neurons * sizeof(double*));
        network->weights[k].rows = network->layers[k].neurons;
        network->weights[k].columns = network->layers[k+1].neurons;

        for (int i = 0; i < network->layers[k].neurons; i++){
            network->weights[k].matrix[i] = (double *) malloc(network->layers[k+1].neurons * sizeof(double));
            fgets(linha, 100*network->layers[k+1].neurons, source);
            network->weights[k].matrix[i][0] = atof(strtok(linha, ","));
            for (int j = 1; j < network->layers[k+1].neurons; j++){
                network->weights[k].matrix[i][j] = atof(strtok(NULL, ","));
            }
        }
    }

    return network;
}

void export_neural_network(neural_network *nn, char *dir) {

    FILE *exported;
    if(dir != NULL){
        exported = fopen(dir, "wt");
    }else{
        exported = fopen("export.txt", "wt");
    }

    fprintf(exported, "%d, %d\n", nn->total_layers, nn->hidden_layers);
    fprintf(exported, "%d, %d\n", nn->input_neurons, nn->output_neurons);
    for (int i = 0; i < nn->total_layers; i++){
        fprintf(exported, "%d, ", nn->layers[i].neurons);
    }
    fprintf(exported, "\n");

    for(int k=0; k < nn->total_layers-1; k++){

        for (int i = 0; i < nn->weights[k].rows; i++){
            for (int j = 0; j < nn->weights[k].columns; j++){
                if(nn->weights[k].matrix == NULL)
                    printf("ERRO EM k = %d, i = %d e j = %d\n", k, i, j);
                fprintf(exported, "%f, ", nn->weights[k].matrix[i][j]);
            }
            fprintf(exported, "\n");
        }
    }

    fclose(exported);
}

// Function to delete a column from the original matrix and substitute it with new columns
double** matrix_fusion(double** original_matrix, int rows, int orig_cols, double** new_matrix, int new_cols, int col_position) {
    // Allocate memory for the new matrix with adjusted column size
    int new_total_cols = orig_cols - 1 + new_cols;
    double** new_result = (double**)malloc(rows * sizeof(double*));
    
    for (int i = 0; i < rows; i++) {
        new_result[i] = (double*)malloc(new_total_cols * sizeof(double));
    }

    // Copy columns before the insertion point
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < col_position; j++) {
            new_result[i][j] = original_matrix[i][j];
        }
    }

    // Insert new columns from the new matrix
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < new_cols; j++) {
            new_result[i][col_position + j] = new_matrix[i][j];
        }
    }

    // Copy remaining columns from the original matrix, skipping the deleted column
    for (int i = 0; i < rows; i++) {
        for (int j = col_position + 1; j < orig_cols; j++) {
            new_result[i][j - 1 + new_cols] = original_matrix[i][j];
        }
    }

    return new_result;
}

int dummy(dataset *data, int column) {
    int *classes = (int*) calloc((data->rows/10),sizeof(int)); //inicia vetor para contabilizar as classes. Assume que há pelo menos 10% de instâncias de cada classe, portanto, 10 classes. Se necessário mais, depois dá o realloc 
    int class_counter = 0;
    for (int i = 0; i < data->rows; i++){
        int unknown_class = 1;
        for (int j = 0; j < class_counter; j++){
            if(data->matrix[i][column] == classes[j])
                unknown_class = 0; //classe ja e conhecida
        }
        if (unknown_class == 1){ //se classe ainda nao e conhecida
            classes[class_counter++] = data->matrix[i][column]; //passa a conhecer
        }
    }

    double **nova_matriz = (double **)malloc(data->rows * sizeof(double));
    for (int i = 0; i < data->rows; i++){
        nova_matriz[i] = (double*)calloc(class_counter, sizeof(double)); //cria uma coluna pra cada classe
    }
    
    for (int i = 0; i < data->rows; i++){
        int current_class = data->matrix[i][column];
        for(int j = 0; j < class_counter; j++){
            nova_matriz[i][j] = (j == current_class ? 1 : 0);
        }
    }


    data->matrix = matrix_fusion(data->matrix, data->rows, data->columns, nova_matriz, class_counter, column);
    data->columns = data->columns - 1 + class_counter;

    return class_counter;
}

void get_dummies(dataset *data) {
    int *indices_para_dummificar = malloc(data->columns * sizeof(int));
    int colunas_counter = 0;
    for (int j = 0; j < data->columns; j++){
        int is_categorical = 1;
        for (int i = 0; i < data->rows; i++){
            if ((double)(int)data->matrix[i][j] != data->matrix[i][j]){ //se for um valor decimal (double) 
                is_categorical = 0;
                break;
            }
        }

        if (is_categorical){ //se todos os valores forem categoricos
            //printf("Coluna %d foi considerada categorica. Chamando dummy:\n", j);
            indices_para_dummificar[colunas_counter++] = j;
        }
    }

    int colunas_adicionadas = 0;
    for (int j = 0; j < colunas_counter; j++){
        //printf("Antes de coluna %d: %d colunas - ", j, data->columns);
        colunas_adicionadas += dummy(data, indices_para_dummificar[j]+colunas_adicionadas) - 1;
        //printf("Depois de coluna %d: %d colunas\n", j, data->columns);
    }
}