CC = gcc
CFLAGS = -I./include -Wall -fPIC

# Alvo para compilar a biblioteca como um arquivo objeto
lib: src/MLP.c src/bamboo.c
	$(CC) $(CFLAGS) -c src/MLP.c -o MLP.o
	$(CC) $(CFLAGS) -c src/bamboo.c -o bamboo.o
	ar rcs libneuralnetworc.a MLP.o bamboo.o  
# Cria uma biblioteca estática (.a)