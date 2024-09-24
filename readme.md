gcc -o executable_name main_file.c library_file.c
./executable_name


Current limitations:
1. The neural network handle only numeric values, as it uses float to represent data. Replace the text fields in your dataset for numeric values before training the neural network with the data. 
2. If you want to remove null values, delete them in your dataset with other libraries before training the neural network. This library treats null values as 0, and that might affect your results. You can turn the "force_not_null" flag on while reading your csv to guarantee your dataset doesn't have any null values.


Usage:
FILE *arquivo; //create variable to store file 
arquivo = fopen("test_data.csv", "r"); //open your file in read mode
dataset *dados = carrega_dados_csv(arquivo, ",", 1000, 1); //send it to be converted to a dataset
escalona_dados(dados); //minmax scaler (if needed)

int camadas_intermediarias = 1; //set hidden layers
camada *camadas, *erros; //create variables to store the layers and the errors
cria_rede_neural(dados, 2, &camadas, camadas_intermediarias, &erros, &rede); //create your neural network

------continua no proximo episodio----------



neural_network * cria_rede_neural(dataset *dados, int neuronios_saida, camada **camadas, int camadas_intermediarias, camada **erros, pesos **rede){ //por enquanto precisa receber neuronios de saida, depois irei fazer a codificacao automatica e o sistema ja ira saber 
    int neuronios = dados->colunas;
    int numero_camadas = camadas_intermediarias + 2;
    *rede = gera_rede_neural(1, neuronios, neuronios_saida, camadas);

    //cria camada para armazenar erros
    *erros = (camada *) malloc(numero_camadas*sizeof(camada));

    //define tamanho de acordo com a rede criada
    for (int i = 0; i < numero_camadas; i++){
        (*erros)[i].neuronios = (*camadas)[i].neuronios;
        (*erros)[i].valor = (float *) calloc((*erros)[i].neuronios, sizeof(float));
    }

    
    for(int i = 0; i < dados->linhas;i++){ //para cada instância
        carrega_camada_entrada(dados->matriz[i] , (*camadas)[0].valor, neuronios); //envia camada de entrada para receber os dados da primeira instancia
        //na versão de treino terá:
        calculate(rede, camadas, numero_camadas);

        float *saida_esperada = (float *) malloc(neuronios_saida * sizeof(float)); //cria vetor pra armazenar saídas
        for (int j = (dados->colunas) - 1; j > (dados->colunas) - 1 - neuronios_saida; j--){
            saida_esperada[(dados->colunas) - 1 - j] = dados->matriz[i][j];
        }
        error_factor(camadas, erros, rede, numero_camadas, saida_esperada);

        // printf("Testes: \n");
        // for (int k = 0; k < 3; k++){
        //     for (int j = 0; j < erros[k].neuronios; j++) 
        //         printf("%.5f ", erros[k].valor[j]);
        //     printf("\n\n");
        // }

        backpropagation(camadas, erros, rede, numero_camadas, 0.9, 0.2);
    }

    neural_network *rede_neural = (neural_network *) malloc(sizeof(rede_neural));
    rede_neural->camadas = (*camadas);
    rede_neural->dados = dados;
    rede_neural->erros = (*erros);
    rede_neural->hidden_layers = camadas_intermediarias;
    rede_neural->output_neurons = neuronios_saida;
    rede_neural->rede = (*rede);
    return rede_neural;
}



neural_network * cria_rede_neural(dataset *dados, int neuronios_saida, camada **camadas, int camadas_intermediarias, camada **erros){ //por enquanto precisa receber neuronios de saida, depois irei fazer a codificacao automatica e o sistema ja ira saber 
    int neuronios = dados->colunas;
    int numero_camadas = camadas_intermediarias + 2;
    pesos * rede = gera_rede_neural(1, neuronios, neuronios_saida, camadas);

    //cria camada para armazenar erros
    *erros = (camada *) malloc(numero_camadas*sizeof(camada));

    //define tamanho de acordo com a rede criada
    for (int i = 0; i < numero_camadas; i++){
        (*erros)[i].neuronios = (*camadas)[i].neuronios;
        (*erros)[i].valor = (float *) calloc((*erros)[i].neuronios, sizeof(float));
    }

    
    for(int i = 0; i < dados->linhas;i++){ //para cada instância
        carrega_camada_entrada(dados->matriz[i] , (*camadas)[0].valor, neuronios); //envia camada de entrada para receber os dados da primeira instancia
        //na versão de treino terá:
        calculate(&rede, camadas, numero_camadas);

        float *saida_esperada = (float *) malloc(neuronios_saida * sizeof(float)); //cria vetor pra armazenar saídas
        for (int j = (dados->colunas) - 1; j > (dados->colunas) - 1 - neuronios_saida; j--){
            saida_esperada[(dados->colunas) - 1 - j] = dados->matriz[i][j];
        }
        error_factor(camadas, erros, &rede, numero_camadas, saida_esperada);

        // printf("Testes: \n");
        // for (int k = 0; k < 3; k++){
        //     for (int j = 0; j < erros[k].neuronios; j++) 
        //         printf("%.5f ", erros[k].valor[j]);
        //     printf("\n\n");
        // }

        backpropagation(camadas, erros, &rede, numero_camadas, 0.9, 0.2);
    }

    neural_network *rede_neural = (neural_network *) malloc(sizeof(rede_neural));
    rede_neural->camadas = (*camadas);
    rede_neural->dados = dados;
    rede_neural->erros = (*erros);
    rede_neural->hidden_layers = camadas_intermediarias;
    rede_neural->output_neurons = neuronios_saida;
    rede_neural->rede = rede;
    return rede_neural;
}