# Verifica se há pelo menos um argumento
if ($args.Count -lt 0) {
    Write-Host "Usage: ./nnc <file(s)> [gcc options]"
    exit 1
}

# Adiciona os arquivos de código fonte adicionais
$additionalFiles = "src/bamboo.c", "src/MLP.c"

# Constrói o comando gcc com os arquivos de entrada, o nome do programa e os arquivos adicionais
$gccCommand = "gcc " + ($args -join " ") +  " " + ($additionalFiles -join " ")

# Executa o comando gcc
Invoke-Expression $gccCommand