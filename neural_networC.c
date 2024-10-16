#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "neural_networC.h"
#define debug 0

float** inicia_matriz(int neuronios, int saida)
{
    //srand(time(NULL));

    float **matriz;
    matriz = (float**)malloc(neuronios*sizeof(float*));
    //printf("Neuronios alocados");
    for(int i = 0; i < neuronios; i++)
        matriz[i] = (float*)malloc(saida*sizeof(float));

    //printf("Saida alocada");
    for(int i = 0; i < neuronios; i++){
        for(int j = 0; j < saida; j++){
            matriz[i][j] = (float) rand() / RAND_MAX;
            // float limit = sqrt(6.0 / (neuronios + saida));
            // matriz[i][j] = ((float)rand() / RAND_MAX) * 2 * limit - limit;


            //printf("%f ", matriz[i][j]);
        }
        //printf("\n");
    }

    return matriz;
}

int desaloca_matriz(int **matriz, int neuronios){
    for(int i = 0; i < neuronios; i++){
        free(matriz[i]);
    }

    free(matriz);
    printf("Matriz desalocada");

    return 0;
}

int verifica_pesos(pesos *rede, int tamanho_rede);

pesos *gera_rede_neural(int n_camadas_intermediarias, int neuronios_entrada, int neuronios_saida, camada **camadas){
    srand(time(0));
    (*camadas) = (camada*)malloc( ( 2 + n_camadas_intermediarias ) * sizeof(camada)); //cria vetor de camadas de neuronios

    //vetores de camadas serão associados à valores do CSV. ISSO AQUI É PARA TESTES
    (*camadas)[0].valor = (float*)malloc(neuronios_entrada * sizeof(float));
    (*camadas)[0].neuronios = neuronios_entrada;
    // for (int i = 0; i < neuronios_entrada; i++){
    //     (*camadas)[0].valor[i] = (float) rand() / RAND_MAX; //coloca valores aleatorios na camada de entrada. APENAS PARA TESTES INICIAIS
    // }
    //###########################################################################################################################

    /*camada de entrada - intermediaria
    intermediaria - camada de saida*/
    pesos *peso_camadas = (pesos*)malloc( ( n_camadas_intermediarias+1 ) *sizeof(pesos)); 

    int neuronios_atuais = neuronios_entrada;
    int neuronios_intermediarios;
    for(int i = 0; i < n_camadas_intermediarias+1; i++){
        if ( i == n_camadas_intermediarias )//se e a ultima camada intermediaria
            neuronios_intermediarios = neuronios_saida; //camada de destino e a de saida
        else
            neuronios_intermediarios = (neuronios_atuais + neuronios_saida) / 2; //atualiza neuronios da camada de destino
        //printf("%d entrada, %d saida", neuronios_atuais, neuronios_intermediarios);

        //###########################################################################################################################
        //i comeca em 0, mas camada 0 ja foi preenchida pela camada de entrada. Precisa agora associar as camadas futuras
        (*camadas)[i+1].valor = (float*)malloc(neuronios_intermediarios * sizeof(float));
        (*camadas)[i+1].neuronios = neuronios_intermediarios;
        //###########################################################################################################################

        peso_camadas[i].matriz = inicia_matriz(neuronios_atuais, neuronios_intermediarios);
        peso_camadas[i].colunas = neuronios_intermediarios;
        peso_camadas[i].linhas = neuronios_atuais;
        neuronios_atuais = neuronios_intermediarios; //atualiza neuronios da camada atual
    }

    verifica_pesos(peso_camadas, 2+n_camadas_intermediarias-1);
    return peso_camadas;
}

float funcao_sigmoide(float x){
    return (float) ( 1.0 / (1.0 + exp( (x * -1) )) );
    //return x;
}

float funcao_relu(float x) {
    return (x < 0) ? 0 : x;
}

int dataframe_head(dataset *data, int head){
    if (head > data->linhas){
        printf("Não ha instancias suficientes\n");
        return 1;
    }
    for (int i = 0; i < head; i++){
        for (int j = 0; j < data->colunas; j++){
            printf("%.4f ", data->matriz[i][j]);
        }
        printf("\n");
    }
    return 0;
}

void mat_mul(float* a, float** b, float* result, int n, int p) {
    // matrix a of size 1 x n (array)
    // matrix b of size n x p
    // matrix result of size 1 x p (array)
    // result = a * b
    int j, k;
    for (j = 0; j < p; j++) {
        result[j] = 0.0;
        for (k = 0; k < n; k++)
            result[j] += (a[k] * b[k][j]);
    }
}

void sigmoid(int n, float* input, float* output) {
    output[0] = 1; // Bias term

    int i;
    for (i = 0; i < n; i++) 
        output[i+1] = 1.0 / (1.0 + exp(-input[i])); // Sigmoid function
}

int calculate(pesos **peso, camada **camadas, int n_camadas){ 
    if (debug) printf("CALCULANDO...:\n");

    //aloca matriz de bias com uma linha para cada camada uma coluna para cada neuronio naquela camada. Resumindo, um bias por neuronio
    float **bias;
    bias = (float**)malloc((n_camadas-1)*sizeof(float*));

    //define valores pro bias (fazer para cada instância ou para cada camada?)
    /*
    layer_outputs[0][0] = 1; // Bias term of input layer
    for (i = 0; i < param->feature_size-1; i++)
        layer_outputs[0][i+1] = layer_inputs[0][i] = param->data_train[training_example][i];
    */
    for (int k = 0; k < n_camadas-1; k++) { //saida nao possui bias
        bias[k] = (float*)malloc((*camadas)[k].neuronios*sizeof(float));
        for (int i = 0; i < (*camadas)[k].neuronios; i++) {
            bias[k][i] = 0.1f; // ou outro valor pequeno, como 0.01
        }
    }

    //minha loucura
    //seta valores para os neuronios
    for(int k=1; k < n_camadas; k++) //zera todas as camadas que não são de entradas de dados (POSSIVEL PROBLEMA. SERA QUE DEVE SER MANTIDO PARA AS PROXIMAS ITERAÇÕES?)
        for (int i = 0 ; i < (*camadas)[k].neuronios; i++)
            (*camadas)[k].valor[i] = 0.0;

    //aplica calculo feedforward
    for(int k=1; k < n_camadas; k++){ //para cada camada, começando da segunda
        for (int i = 0 ; i < (*camadas)[k].neuronios; i++){ //neuronios da camada atual
            if (debug) printf("Neuronio atual (camada %d, neuronio %d):\n", k, i);
            for (int j = 0 ; j < (*camadas)[k-1].neuronios; j++){ //neuronios da camada anterior
                if (debug) printf("Valor anterior (%f) * peso (%f) = %f. Valor do neuronio (%f) + resultado (%f) = ",
                 (*camadas)[k-1].valor[j], (*peso)[k-1].matriz[j][i], (*camadas)[k-1].valor[j] * (*peso)[k-1].matriz[j][i],
                 (*camadas)[k].valor[i], (*camadas)[k-1].valor[j] * (*peso)[k-1].matriz[j][i]);

                (*camadas)[k].valor[i] += (*camadas)[k-1].valor[j] * (*peso)[k-1].matriz[j][i]; //camada atual recebe += valor * peso da conexao com camada anterior
                if (debug) printf("%f\n", (*camadas)[k].valor[i]);
            }

            //a ultima camada está sempre saindo com um valor positivo e um negativo
            // if(k == n_camadas-1)
            //     printf("O valor é: %f", (*camadas)[k].valor[i]);

            (*camadas)[k].valor[i] = funcao_sigmoide((*camadas)[k].valor[i]); //aplica funcao sigmoide - + bias[k-1][i]
            if (debug) printf("Apos sigmoide: %f\n", (*camadas)[k].valor[i]);

            // if(k == n_camadas-1)
            //     printf(", mas virou: %f\n", (*camadas)[k].valor[i]);
        }
    }

    /*loucura copiada do cõdigo alheio
    int i;
    bias[0][0] = 1; // Bias term of input layer
    for (i = 0; i < (*camadas)[0].neuronios; i++)
        bias[0][i] = (*camadas)[0].valor[i];

    // Perform forward propagation for each hidden layer
    // Calculate input and output of each hidden layer
    for (i = 1; i < n_camadas; i++) {
        // Compute layer_inputs[i]
        mat_mul((*camadas)[i-1].valor, (*peso)[i-1].matriz, (*camadas)[i].valor, (*camadas)[i-1].neuronios, (*camadas)[i].neuronios); //(*camadas)[i-1].neuronios+1

        // Compute layer_outputs[i]
        // Activation functions (identity - 1, sigmoid - 2, tanh - 3, relu - 4, softmax - 5)
        sigmoid((*camadas)[i].neuronios, (*camadas)[i].valor, (*camadas)[i].valor);
    }*/


    /* nao sei em que momento usar esse threshold, acho que so na previsao, nao no treinamento
    for (int i = 0; i < (*camadas)[n_camadas-1].neuronios; i++){ //para cada neuronio na camada de saida
        if ( (*camadas)[n_camadas-1].valor[i] > 0.5) //usa o threshold para transformar em 0 ou 1, respeitando a codificação
            (*camadas)[n_camadas-1].valor[i] = 1;
        else
            (*camadas)[n_camadas-1].valor[i] = 0;
    }
    */
    return 0; //camadas ja contem os resultados. pensar se haverá retorno ou se vouusar o próprio camadas
}

dataset *carrega_dados_csv(FILE* arquivo, char *delim, int tam_max_linha, short force_not_null){
    char linha[tam_max_linha];  // Para armazenar uma linha do CSV
    char *token;            // Para armazenar cada célula da linha

    // Verifica se o arquivo foi aberto corretamente
    if (arquivo == NULL) {
        printf("Erro ao abrir o arquivo!\n");
        return NULL;
    }

    //printf("Passou!\n");

    int num_colunas = 0, num_linhas = 0;
    fgets(linha, sizeof(linha), arquivo); //le o header - aqui, PRECISA ter header
    token = strtok(linha, delim);
    while (token != NULL) {
        num_colunas++; //conta quantas colunas possui 
        token = strtok(NULL, delim);
    }
    while (fgets(linha, sizeof(linha), arquivo)) {
        num_linhas++; //conta quantas linhas
    }
    //printf("Passou!\n");

    dataset *dados = (dataset*) malloc(sizeof(dataset));
    dados->colunas = num_colunas;
    dados->linhas = num_linhas;
    dados->matriz = (float**)malloc(num_linhas*sizeof(float*));
    for (int i = 0; i < num_linhas; i++){
        dados->matriz[i] = (float*)calloc(num_colunas, sizeof(float));
    }
    //printf("Passou!\n");

    rewind(arquivo); //volta para o comeco
    fgets(linha, sizeof(linha), arquivo); //ignora header
    
    //printf("Passou!\n");

    int i = 0, j = 0;
    // Lê o arquivo linha por linha
    while (fgets(linha, sizeof(linha), arquivo)) {
        
        // Remove o '\n' no final da linha, se houver
        linha[strcspn(linha, "\n")] = '\0';
        // printf("%s - %d %d\n", linha, num_linhas, num_colunas);

        // Divide a linha em colunas usando strtok
        token = strtok(linha, delim);
        while (token != NULL) {
            //printf("%s ", token);  // Exibe cada valor da coluna
            dados->matriz[i][j] = atof(token);
            token = strtok(NULL, delim);            // Pega o próximo valor
            if (token == NULL && j < num_colunas-1) { //verifica se é um valor nulo e não no fim da linha
                printf("ERRO: Valor nulo não esperado encontrado na linha %d, campo %d\n", i, j);
                if(force_not_null) //se o usuario pediu para encerrar o programa ao encontrar valores nulos, encerra
                    exit(1);
            }
            j++;
        }

        //printf("\n");  // Quebra de linha após cada linha do CSV
        j = 0;
        i++;
    }
    //printf("Passou!\n");

    // Fecha o arquivo
    fclose(arquivo);

    return dados;
}

int escalona_dados(dataset *dados){
    float **matriz = dados->matriz;

    //percorre uma coluna de cada vez, encontrando maior e menor valor
    for(int j = 0; j < dados->colunas; j++){
        float maior = matriz[0][j];
        float menor = matriz[0][j];

        for(int i = 1; i < dados->linhas; i++){
            if (matriz[i][j] > maior)
                maior =  matriz[i][j];
            else if (matriz[i][j] < menor)
                menor =  matriz[i][j];
        }

        //uma vez encontrados os maiores e menores valores, aplica escalonamento minmax
        for(int i = 0; i < dados->linhas; i++){
            matriz[i][j] = (matriz[i][j] - menor) / (maior - menor);  
        }
    }

    return 0;
}

int carrega_camada_entrada(float *dados, float *camada, int tamanho){
    for (int i = 0; i < tamanho; i++){
        camada[i] = dados[i];
    }
}

// Função para inicializar delta
pesos *inicializar_delta(pesos *pesos_rede, int num_camadas) {

    pesos *deltas = (pesos*)malloc( ( num_camadas - 1 ) *sizeof(pesos)); 
    for (int k = 0; k < num_camadas - 1; k++) {
        deltas[k].linhas = pesos_rede[k].linhas;
        deltas[k].colunas = pesos_rede[k].colunas;
        
        // Aloca a matriz de deltas com os mesmos tamanhos da matriz de pesos
        deltas[k].matriz = (float**) malloc(deltas[k].linhas * sizeof(float*));
        for (int i = 0; i < deltas[k].linhas; i++) {
            deltas[k].matriz[i] = (float*) malloc(deltas[k].colunas * sizeof(float));
            // Inicializa com 0.0
            for (int j = 0; j < deltas[k].colunas; j++) {
                deltas[k].matriz[i][j] = 0.0;
            }
        }
    }

    return deltas;
}

//vai virar treina rede
neural_network * cria_rede_neural(dataset *dados, int neuronios_saida, int camadas_intermediarias){ //por enquanto precisa receber neuronios de saida, depois irei fazer a codificacao automatica e o sistema ja ira saber 
    int neuronios = dados->colunas;
    int numero_camadas = camadas_intermediarias + 2;
    camada *camadas;
    pesos * rede = gera_rede_neural(camadas_intermediarias, neuronios, neuronios_saida, &camadas);


    // printf("REDE GERADA: \n");
    // for (int i = 0; i < 5; i++){ //rede->linhas
    //     for (int j = 0; j < rede->colunas; j++)
    //         printf("%f ", rede->matriz[i][j]);
    //     printf("\n");
    // }
    //cria camada para armazenar erros
    camada *erros = (camada *) malloc(numero_camadas*sizeof(camada));

    //define tamanho de acordo com a rede criada
    for (int i = 0; i < numero_camadas; i++){
        erros[i].neuronios = camadas[i].neuronios;
        erros[i].valor = (float *) calloc(erros[i].neuronios, sizeof(float));
    }

    pesos *delta = inicializar_delta(rede, numero_camadas);

    neural_network *rede_neural = (neural_network *) malloc(sizeof(neural_network));
    rede_neural->camadas = camadas;
    rede_neural->dados = dados;
    rede_neural->erros = erros;
    rede_neural->hidden_layers = camadas_intermediarias;
    rede_neural->output_neurons = neuronios_saida;
    rede_neural->rede = rede;
    rede_neural->delta = delta;
    return rede_neural;
}

int verifica_erro(camada *erro){
    printf("Erros: \n");
    for (int i = 1; i < 3; i++){ //camada 0 vai ficar zerada pois entradas não tem erro
        for (int j = 0; j < erro[i].neuronios; j++) 
            printf("%.5f ", erro[i].valor[j]);
        printf("\n");
    }
}

int verifica_camadas(camada *camada){
    printf("Camadas: \n");
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < camada[i].neuronios; j++) 
            printf("%.5f ", camada[i].valor[j]);
        printf("\n");
    }
}

int x_y_split(dataset *original, dataset **x, dataset **y, int output_neurons){
    // Alocando memória para a matriz Y
    int rows = original->linhas;
    *y = (dataset *)malloc(sizeof(dataset));
    (*y)->linhas = rows;
    (*y)->colunas = output_neurons;
    (*y)->matriz = (float **)malloc(rows * sizeof(float *));
    for (int i = 0; i < rows; i++) {
        (*y)->matriz[i] = (float *)malloc(output_neurons * sizeof(float));
    }

    // Alocando memória para a matriz X
    int remaining_cols = original->colunas - output_neurons;
    *x = (dataset *)malloc(sizeof(dataset));
    (*x)->linhas = rows;
    (*x)->colunas = remaining_cols;
    (*x)->matriz = (float **)malloc(rows * sizeof(float *));
    for (int i = 0; i < rows; i++) {
        (*x)->matriz[i] = (float *)malloc(remaining_cols * sizeof(float));
    }

    // Copiando os dados para y e x
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < remaining_cols; j++) {
            (*x)->matriz[i][j] = original->matriz[i][j];  // Copia para X
        }
        for (int j = 0; j < output_neurons; j++) {
            (*y)->matriz[i][j] = original->matriz[i][j + remaining_cols];  // Copia para Y
        }
    }
}

int verifica_pesos(pesos *rede, int tamanho_rede){
    //printf("REDE GERADA com %d camadas", tamanho_rede);
    for( int k = 0; k < tamanho_rede; k++){
        //printf("com %d linhas e %d colunas: \n", rede[k].linhas, rede[k].colunas);
        for (int i = 0; i < rede[k].linhas; i++){ //rede->linhas
            for (int j = 0; j < rede[k].colunas; j++)
                printf("%f ", rede[k].matriz[i][j]);
            printf("\n");
        }
        printf("\n");
    }
}

int treina_rede_neural(neural_network *nn, dataset *saidas_esperadas, int epocas, float momentum, float alpha){
    verifica_pesos(nn->delta, nn->hidden_layers+1);
    int neuronios = nn->dados->colunas;
    int numero_camadas = (nn->hidden_layers) + 2;
    for(int e = 0; e < epocas; e++){
        if (debug) printf("\n----------Nova epoca %d----------\n", e);
        for(int i = 0; i < nn->dados->linhas;i++){ //para cada instância
            carrega_camada_entrada(nn->dados->matriz[i] , nn->camadas[0].valor, neuronios); //envia camada de entrada para receber os dados da primeira instancia
            
            // if( i < 3){
            //     printf("Camadas: \n");
            //     for (int k = 0; k < nn->hidden_layers + 2; k ++){
            //         for (int j = 0; j < nn->camadas[k].neuronios; j ++){
            //             printf("%f ", nn->camadas[k].valor[j]);
            //         }
            //         printf("\n");
            //     }
            //     printf("Esperado: ");

            //     for (int j = 0; j < saidas_esperadas->colunas; j ++){
            //         printf("%f ", saidas_esperadas->matriz[i][j]);
            //     }
            //     printf("\n");
            // }            
            
            //na versão de treino terá:
            calculate(&(nn->rede), &(nn->camadas), numero_camadas);

            // float *saida_esperada = (float *) malloc(nn->output_neurons * sizeof(float)); //cria vetor pra armazenar saídas
            // for (int j = (nn->dados->colunas) - 1; j > (nn->dados->colunas) - 1 - nn->output_neurons; j--){
            //     saida_esperada[(nn->dados->colunas) - 1 - j] = nn->dados->matriz[i][j];
            // }
            float *saida_esperada = saidas_esperadas->matriz[i];
            // verifica_camadas(nn->camadas);
            // verifica_erro(nn->erros);
            //if( i < 3) verifica_pesos(nn->rede);
            backpropagation(&(nn->camadas), &(nn->erros), &(nn->rede), &(nn->delta), saida_esperada, numero_camadas, momentum, alpha);
            if (debug) printf("\n\n");
        }
    }
}

//calcula os erros da iteração atual
int error_factor(camada **camadas, camada **camada_erros, pesos **rede, int num_camadas, float *saida_esperada){ //camada_erros mantem os erros de cada neuronio ao inves de seu valor
    if (debug) printf("ERRANDO...:\n");
    
    //para a camada de saida
    for(int i = 0; i < (*camadas)[num_camadas-1].neuronios; i++){ //para cada neuronio da camada de saida
        if (debug) printf("Neuronio atual (camada %d, neuronio %d):\n", num_camadas-1, i);
        float saida_neuronio = (*camadas)[num_camadas-1].valor[i]; //saida do neuronio sendo inspecionado
        //printf("%.2f - %.2f   ", saida_esperada[i], saida_neuronio);
        
        float fator_erro = saida_esperada[i] - saida_neuronio; //diferenca da saida que esperava com a saida que recebeu
        float erro = saida_neuronio * (1 - saida_neuronio) * fator_erro; //calcula erro

        (*camada_erros)[num_camadas-1].valor[i] = erro; //passa erro para camada de erros
        if (debug) printf("Saida esperada (%f) - saida recebida (%f) = %f. Saida recebida (%f) * (1 - saida recebida) (%f) * fator erro (%f) = erro (%f)\n"
        , saida_esperada[i], saida_neuronio, fator_erro, saida_neuronio, 1-saida_neuronio, fator_erro, erro);
    }
    //printf("\n");

    //para camadas intermediarias
    for(int k = num_camadas-2; k > 0; k--){ //comeca da camada anterior à saida e vai voltando até o fim, camada de entrada nao tem erro (se tiver, >=)
        for (int i = 0; i < (*camadas)[k].neuronios; i++){ //para cada neuronio dessa camada
        if (debug) printf("Neuronio atual (camada %d, neuronio %d):\n", k, i);
            float fator_erro = 0;
            for (int j = 0; j < (*camadas)[k+1].neuronios; j++){ //verifica quais neuronios da camada seguinte se conetam a ele
                float erro_seguinte = (*camada_erros)[k+1].valor[j]; //erro do neuronio da camada seguinte
                fator_erro += erro_seguinte * (*rede)[k].matriz[i][j]; //soma os erros ponderados dos neuronios na camada seguinte
            }    

            float saida_neuronio = (*camadas)[k].valor[i]; //saida do neuronio sendo inspecionado
            float erro = saida_neuronio * (1 - saida_neuronio) * fator_erro; //calcula erro

            (*camada_erros)[k].valor[i] = erro; //passa erro para camada de erros
        }
    }
}

float get_error(camada **camadas, camada **camada_erros, pesos **rede, int camada, int num_camadas, int neuronio, float saida_esperada){
    if (debug) printf("ERRANDO...:\n");
    
    if (num_camadas-1 == camada){ //para camada de saida
        float saida_neuronio = (*camadas)[camada].valor[neuronio]; //saida do neuronio sendo inspecionado
        //printf("%.2f - %.2f   ", saida_esperada[i], saida_neuronio);
        
        float fator_erro = saida_esperada - saida_neuronio; //diferenca da saida que esperava com a saida que recebeu
        float erro = saida_neuronio * (1 - saida_neuronio) * fator_erro; //calcula erro
        (*camada_erros)[camada].valor[neuronio] = erro;
         if (debug) printf("Saida esperada (%f) - saida recebida (%f) = %f. Saida recebida (%f) * (1 - saida recebida) (%f) * fator erro (%f) = erro (%f)\n"
        , saida_esperada, saida_neuronio, fator_erro, saida_neuronio, 1-saida_neuronio, fator_erro, erro);

        return erro;
    }else{ //para camadas intermediarias
        float fator_erro = 0;
                for (int j = 0; j < (*camadas)[camada+1].neuronios; j++){ //verifica quais neuronios da camada seguinte se conetam a ele
                    float erro_seguinte = (*camada_erros)[camada+1].valor[j]; //erro do neuronio da camada seguinte
                    fator_erro += erro_seguinte * (*rede)[camada].matriz[neuronio][j]; //soma os erros ponderados dos neuronios na camada seguinte
                }    

                float saida_neuronio = (*camadas)[camada].valor[neuronio]; //saida do neuronio sendo inspecionado
                float erro = saida_neuronio * (1 - saida_neuronio) * fator_erro; //calcula erro

                (*camada_erros)[camada].valor[neuronio] = erro; //passa erro para camada de erros
        if (debug) printf("Erro calculado = %f\n", erro);

                return erro;
    }
}


//REVISAR TODO O BACKPROPAGATION E TODA A INICIALIZAÇÃO DOS DEMAIS ELEMENTOS - PERCORRE TODAS AS DEMAIS ESTRUTURAS E PRINTA PRA VERIFICAR
int backpropagation(camada **camadas, camada **erro, pesos **peso, pesos **previous_delta, float *saida_esperada, int num_camadas, float momentum, float alpha ) { 
    if (debug) printf("RETROPROPAGANDO...\n");
    
    for (int k = num_camadas - 1; k > 0; k--) { // from output layer back to input layer
        for (int j = 0; j < (*peso)[k-1].colunas; j++) { // iterate through neurons of the current layer
            // Calculate the error for the neuron
            float erro_neuronio = get_error(camadas, erro, peso, k, num_camadas, j, saida_esperada[j]); 
            
            // Update each weight connecting from the previous layer to this neuron
            for (int i = 0; i < (*peso)[k-1].linhas; i++) { 
                if (debug) printf("Neuronio atual (camada %d, neuronio anterior %d, neuronio posterior %d):\n", k, i, j);
                
                // Correct weight update (adjusting from the previous layer to this neuron)
                if (debug) printf("Peso (%f) + Peso_anterior (%f) * momentum (%f) + alpha (%f) * saida neuronio anterior (%f) * erro neuronio posterior (%f) = ",
                                  (*peso)[k-1].matriz[i][j], (*previous_delta)[k-1].matriz[i][j], momentum, alpha, (*camadas)[k-1].valor[i], erro_neuronio);

                // Apply weight update using momentum and learning rate
                //(*peso)[k-1].matriz[i][j] = (*peso)[k-1].matriz[i][j] * momentum + alpha * (*camadas)[k-1].valor[i] * erro_neuronio;
                //(*peso)[k-1].matriz[i][j] += alpha * (*camadas)[k-1].valor[i] * erro_neuronio;

                // Assuming you store previous weight updates in a matrix 'previous_delta'
                float weight_update = momentum * (*previous_delta)[k-1].matriz[i][j] + alpha * (*camadas)[k-1].valor[i] * erro_neuronio;
                (*previous_delta)[k-1].matriz[i][j] = weight_update;  // Store for momentum in the next step
                (*peso)[k-1].matriz[i][j] += weight_update;  // Apply the weight update

                
                if (debug) printf("%f\n", (*peso)[k-1].matriz[i][j]);
            }
        }
    }
    return 0;
}
