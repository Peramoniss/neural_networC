#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "neural_networC.h"

int main()
{
    FILE *arquivo;
    arquivo = fopen("test_data_2.csv", "r");
    dataset *dados = carrega_dados_csv(arquivo, ",", 1000, 1);
    int neuronios_saida = 2;

    //escalona_dados(dados); //altera na própria estrutura
    
    dataframe_head(dados, 4);

    dataset *x, *y;
    x_y_split(dados, &x, &y, neuronios_saida);

    //dataframe_head(x, 4);
    //dataframe_head(y, 4);



    //LOOP CARREGANDO DADOS DE OUTRAS LINHAS DO CSV PARA A PRIMEIRA CAMADA. VAI ALTERANDO OS PESOS E OS SALVANDO
    //RESULTADO DA ULTIMA CAMADA É USADO PARA CALCULAR ERRO - PRECISO VER COMO USAREI O ERRO E O BACKPROPAGATION
    //FAZER TRAIN TEST SPLIT ANTES DISSO
        //usuario deve saber o numero de camadas. Se trata da de entrada + saida + intermediarias, definidas no primeiro parametro de gera_rede_neural
    /*camada *camadas = (camada*) malloc(sizeof(camada));
    pesos *rede = (pesos *) malloc(sizeof(pesos));*/

    int camadas_intermediarias = 1;
    camada *camadas, *erros;
    //camadas = (camada*) malloc((camadas_intermediarias+2) * sizeof(camada));
    printf("\n\n\nGera rede");
    //pesos *rede; // = gera_rede_neural(camadas_intermediarias, dados->colunas, 2, &camadas);

    printf("Cria rede");
    neural_network *nn;

    nn = cria_rede_neural(x, neuronios_saida, camadas_intermediarias); //PRECISA SEPARAR CRIACAO DO TREINAMENTO, E DEPOIS RODAR O
    //TREINAMENTO N VEZES PARA FUNCIONAR O SISTEMA DE ÉPOCAS. DAÍ TESTA PARA VER SE ESSE É O PROBLEMA
    printf("Criada");

    dataframe_head(y, 5);

    treina_rede_neural(nn, y, 1000); 
    printf("Treinada");

    // for (int i = 0; i < 2; i++){
    //     for(int j = 0; j < rede[i].linhas; j++){
    //         for(int k = 0; k < rede[i].colunas; k++){
    //             printf("%.2f ", rede[i].matriz[j][k]);
    //         }
    //     printf("\n\n");
    //     }
    //     printf("\n\n\n");
    // }

    float *teste = dados->matriz[0];
    for (int i = 0; i < dados->linhas; i++){ //dados->linhas
        carrega_camada_entrada(dados->matriz[i], nn->camadas[0].valor, nn->camadas[0].neuronios); //testa a predicao com primeira instancia
        pesos *rede_temp = nn->rede;
        calculate(&(nn->rede), &(nn->camadas), camadas_intermediarias+2); //faz o calculo
        printf("Teste de previsao: ");

        for (int j = 0; j < nn->camadas[camadas_intermediarias+2-1].neuronios; j++){
            printf("Classe %d: %f (%f), ", j, nn->camadas[camadas_intermediarias+2-1].valor[j], y->matriz[i][j]);
        }
            //printf("- Esperado: %f ", nn->camadas[camadas_intermediarias+2-1].valor[i]);
        printf("instancia: %d\n", i);
    }
        
        

    // printf("Testes: \n");
    // for (int i = 0; i < 3; i++){
    //     for (int j = 0; j < erros[i].neuronios; j++) 
    //         printf("%.5f ", erros[i].valor[j]);
    //     printf("\n");
    // }

    return 0;
}

/*REPRESENTACAO DOS DADOS:
* LE ARQUIVO E LE A PRIMEIRA LINHA - header
* NO HEADER, faz strtok e vai contando até nao encontrar mais virgula - indica o numero de colunas
* depois, vai lendo ate o final e contando o numero de linhas - consegue as linhas
* cria matriz do tamanho encontrado e percorre tudo de novo, salvando os valores 
* talvez tenha que manter o numero de colunas que representam o alvo? Quando os dados entram, tudo certo, mas quando passam pelo ONE HOT ENCODER esse controle vai ser necessario
*/

//DADOS DO CSV JA ESTAO EM MATRIZ

/* AGORA, PARA CADA COLUNA, ENVIA O INDICE PARA UMA FUNCAO QUE IRA VERIFICAR SE OS DADOS SAO CATEGORICOS OU CONTINUOS
* SERA CATEGORICO SE NÃO HOUVER NENHUM VALOR FLUTUANTE - float (resto de divisao?). PERCORRE TODA A COLUNA PARA VERIFICAR, E VAI ADICIONANDO COLUNA EM VETOR
* SE FOR CATEGORICO, ENVIA O VETOR DA COLUNA PARA TRANSFORMAR EM MATRIZ CODIFICADA COM ONE-HOT-ENCODER/DUMMY (JA PROTOTIPADO POR GPT)
* DEPOIS, SUBSTITUI A COLUNA ORIGINAL PELA MATRIZ CODIFICADA (JA PROTOTIPADO POR GPT) - CUIDAR PARA ATUALIZAR A STRUCT COM NOVO TAMANHO DA COLUNAS
* CONTINUA REPETINDO PARA CADA COLUNA E FICA COM MATRIZ CAPAZ DE REPRESENTAR TODOS OS DADOS
*/