gcc -o executable_name main_file.c library_file.c
./executable_name


Current limitations:
1. The neural network handle only numeric values, as it uses float to represent data. Since atof function is used, anything that's non-numeric will stop the reading proccess, and if the field starts with a character, it'll be treated as 0.0. Replace the text fields in your dataset for numeric values before training the neural network with the data for the expected results. 
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