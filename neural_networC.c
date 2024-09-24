#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "neural_networC.h"

float** inicia_matriz(int neuronios, int saida)
{
    srand(time(NULL));

    float **matriz;
    matriz = (float**)malloc(neuronios*sizeof(float*));
    //printf("Neuronios alocados");
    for(int i = 0; i < neuronios; i++)
        matriz[i] = (float*)malloc(saida*sizeof(float));

    //printf("Saida alocada");
    for(int i = 0; i < neuronios; i++){
        for(int j = 0; j < saida; j++){
            matriz[i][j] = (float) rand() / RAND_MAX * 0.01;
            //printf("%d ", matriz[i][j]);
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

pesos *gera_rede_neural(int n_camadas_intermediarias, int neuronios_entrada, int neuronios_saida, camada **camadas){
    srand(time(NULL));
    (*camadas) = (camada*)malloc( ( 2 + n_camadas_intermediarias ) * sizeof(camada)); //cria vetor de camadas de neuronios

    //vetores de camadas serão associados à valores do CSV. ISSO AQUI É PARA TESTES
    (*camadas)[0].valor = (float*)malloc(neuronios_entrada * sizeof(float));
    (*camadas)[0].neuronios = neuronios_entrada;
    for (int i = 0; i < neuronios_entrada; i++){
        (*camadas)[0].valor[i] = (float) rand() / RAND_MAX; //coloca valores aleatorios na camada de entrada. APENAS PARA TESTES INICIAIS
    }
    //###########################################################################################################################

    pesos *peso_camadas = (pesos*)malloc(n_camadas_intermediarias*sizeof(pesos)); //não seria n_camadas+1?

    int neuronios_atuais = neuronios_entrada;
    int neuronios_intermediarios;
    for(int i = 0; i < n_camadas_intermediarias+1; i++){
        if ( i == n_camadas_intermediarias )//se e a ultima camada intermediaria
            neuronios_intermediarios = neuronios_saida; //camada de destino e a de saida
        else
            neuronios_intermediarios = (neuronios_atuais + neuronios_saida) / 2; //atualiza neuronios da camada de destino

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

    return peso_camadas;
}

float funcao_sigmoide(float x){
    return (float) (1.0 / (1.0 + exp(-x)));
}

int calculate(pesos **peso, camada **camadas, int n_camadas){ 

    //aloca matriz de bias com uma linha para cada camada uma coluna para cada neuronio naquela camada. Resumindo, um bias por neuronio
    float **bias;
    bias = (float**)malloc((n_camadas-1)*sizeof(float*));

    //define valores pro bias
    for (int k = 0; k < n_camadas-1; k++) { //saida nao possui bias
        bias[k] = (float*)malloc((*camadas)[k].neuronios*sizeof(float));
        for (int i = 0; i < (*camadas)[k].neuronios; i++) {
            bias[k][i] = 0.1f; // ou outro valor pequeno, como 0.01
        }
    }

    //seta valores para os neuronios
    for(int k=1; k < n_camadas; k++) //zera todas as camadas que não são de entradas de dados
        for (int i = 0 ; i < (*camadas)[k].neuronios; i++)
            (*camadas)[k].valor[i] = 0;

    //aplica calculo feedforward
    for(int k=1; k < n_camadas; k++){ //para cada camada, começando da segunda
        for (int i = 0 ; i < (*camadas)[k].neuronios; i++){ //neuronios da camada atual
            for (int j = 0 ; j < (*camadas)[k-1].neuronios; j++){ //neuronios da camada anterior
                (*camadas)[k].valor[i] += (*camadas)[k-1].valor[j] * (*peso)[k-1].matriz[j][i]; //camada atual recebe valor * peso da conexao com camada anterior
            
            }

            (*camadas)[k].valor[i] = funcao_sigmoide((*camadas)[k].valor[i] + bias[k-1][i]); //aplica funcao sigmoide
        }
    }

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

//vai virar treina rede
neural_network * cria_rede_neural(dataset *dados, int neuronios_saida, int camadas_intermediarias){ //por enquanto precisa receber neuronios de saida, depois irei fazer a codificacao automatica e o sistema ja ira saber 
    int neuronios = dados->colunas;
    int numero_camadas = camadas_intermediarias + 2;
    camada *camadas;
    pesos * rede = gera_rede_neural(1, neuronios, neuronios_saida, &camadas);

    //cria camada para armazenar erros
    camada *erros = (camada *) malloc(numero_camadas*sizeof(camada));

    //define tamanho de acordo com a rede criada
    for (int i = 0; i < numero_camadas; i++){
        erros[i].neuronios = camadas[i].neuronios;
        erros[i].valor = (float *) calloc(erros[i].neuronios, sizeof(float));
    }

    
    for(int i = 0; i < dados->linhas;i++){ //para cada instância
        carrega_camada_entrada(dados->matriz[i] , camadas[0].valor, neuronios); //envia camada de entrada para receber os dados da primeira instancia
        //na versão de treino terá:
        calculate(&rede, &camadas, numero_camadas);

        float *saida_esperada = (float *) malloc(neuronios_saida * sizeof(float)); //cria vetor pra armazenar saídas
        for (int j = (dados->colunas) - 1; j > (dados->colunas) - 1 - neuronios_saida; j--){
            saida_esperada[(dados->colunas) - 1 - j] = dados->matriz[i][j];
        }
        error_factor(&camadas, &erros, &rede, numero_camadas, saida_esperada);

        // printf("Testes: \n");
        // for (int k = 0; k < 3; k++){
        //     for (int j = 0; j < erros[k].neuronios; j++) 
        //         printf("%.5f ", erros[k].valor[j]);
        //     printf("\n\n");
        // }

        backpropagation(&camadas, &erros, &rede, numero_camadas, 0.9, 0.2);
    }

    neural_network *rede_neural = (neural_network *) malloc(sizeof(neural_network));
    rede_neural->camadas = camadas;
    rede_neural->dados = dados;
    rede_neural->erros = erros;
    rede_neural->hidden_layers = camadas_intermediarias;
    rede_neural->output_neurons = neuronios_saida;
    rede_neural->rede = rede;
    return rede_neural;
}

//calcula os erros da iteração atual
int error_factor(camada **camadas, camada **camada_erros, pesos **rede, int num_camadas, float *saida_esperada){ //camada_erros mantem os erros de cada neuronio ao inves de seu valor
    //sprintf("Errando...");
    float fator_erro_x = 0;
    
    //para a camada de saida
    for(int i = 0; i < (*camadas)[num_camadas-1].neuronios; i++){ //para cada neuronio da camada de saida
        float saida_neuronio = (*camadas)[num_camadas-1].valor[i]; //saida do neuronio sendo inspecionado
        //printf("%.2f - %.2f   ", saida_esperada[i], saida_neuronio);
        float fator_erro = saida_esperada[i] - saida_neuronio; //diferenca da saida que esperava com a saida que recebeu
        float erro = saida_neuronio * (1 - saida_neuronio) * fator_erro; //calcula erro

        (*camada_erros)[num_camadas-1].valor[i] = erro; //passa erro para camada de erros
    }
    //printf("\n");

    //para camadas intermediarias
    for(int k = num_camadas-2; k > 0; k--){ //comeca da camada anterior à saida e vai voltando até o fim, camada de entrada nao tem erro (se tiver, >=)
        for (int i = 0; i < (*camadas)[k].neuronios; i++){ //para cada neuronio dessa camada
            float fator_erro = 0;
            for (int j = 0; j < (*camadas)[k+1].neuronios; j++){ //verifica quais neuronios da camada seguinte se conetam a ele
                float erro_seguinte = (*camada_erros)[k+1].valor[j]; //erro do neuronio da camada seguinte
                fator_erro += erro_seguinte * (*rede)[k].matriz[i][j]; //soma os erros ponderados dos neuronios na camada seguinte
            }    

            float saida_neuronio = (*camadas)[k].valor[i]; //saida do neuronio sendo inspecionado
            float erro = saida_neuronio * (1 - saida_neuronio) + fator_erro; //calcula erro

            (*camada_erros)[k].valor[i] = erro; //passa erro para camada de erros
        }
    }
}

//REVISAR TODO O BACKPROPAGATION E TODA A INICIALIZAÇÃO DOS DEMAIS ELEMENTOS - PERCORRE TODAS AS DEMAIS ESTRUTURAS E PRINTA PRA VERIFICAR
int backpropagation(camada **camadas, camada **erro, pesos **rede, int num_camadas, float momentum, float alpha ){ //alpha = taxa de aprendizagem
    for(int k = num_camadas-2; k >= 0; k--){ //comeca da camada anterior a saida e vai voltando até o fim 
        //printf("Moleza %d - %d linhas", k, (*rede)[k].linhas);
        for(int i = 0; i < (*rede)[k].linhas; i++){ //para cada neuronio da camada de entrada em relacao aos pesos
            //printf("Barbada %d", (*rede)[k].colunas);
            for(int j = 0; j < (*rede)[k].colunas; j++){ //verifica cada neuronio da camada de saida em relacao aos pesos
                //calcula novo peso para aquela conexao
                // printf("CALCULANDO");
                // printf("Rede: %.2f\n", (*rede)[k].matriz[i][j]);
                // printf("momentum: %.2f\n", momentum);
                // printf("Saida anterior: %.2f\n", (*camadas)[k].valor[i]);
                // printf("Erro posterior: %.2f\n", (*erro)[k+1].valor[j]);

                (*rede)[k].matriz[i][j] = (*rede)[k].matriz[i][j] * momentum + alpha * (*camadas)[k].valor[i] * (*erro)[k+1].valor[j];//saida da camada anterior * erro da posterior                
                //(*rede)[k].matriz[i][j] += alpha * (*erro)[k+1].valor[j] * (*camadas)[k].valor[i];

            }
            //printf("\n");
        }
            //printf("\n");

    }
}