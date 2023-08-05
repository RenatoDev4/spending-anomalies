import torch

DATA_PATH = "data/processed/cota-parlamentar.csv"
COLUNA_VALOR = 'vlrdocumento'
COLUNA_NOME_PARLAMENTAR = 'txnomeparlamentar'
COLUNA_FORNECEDOR = 'txtfornecedor'
ATIVIDADE_PARLAMENTAR = 'txtdescricao'
TOP = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_SIZE = 0.3
RANDOM_STATE = 42
BATCH_SIZE = 256

TEXTO_BUSCAR = {
    'DIVULGAÇÃO DA ATIVIDADE PARLAMENTAR.': 0,
    'FORNECIMENTO DE ALIMENTAÇÃO DO PARLAMENTAR': 1,
    'MANUTENÇÃO DE ESCRITÓRIO DE APOIO À ATIVIDADE PARLAMENTAR': 2,
    'COMBUSTÍVEIS E LUBRIFICANTES.': 3,
    'PASSAGENS AÉREAS': 4,
    'TELEFONIA': 5,
    'LOCAÇÃO OU FRETAMENTO DE VEÍCULOS AUTOMOTORES': 6,
    'SERVIÇOS POSTAIS': 7,
    'CONSULTORIAS, PESQUISAS E TRABALHOS TÉCNICOS.': 8,
    'SERVIÇO DE SEGURANÇA PRESTADO POR EMPRESA ESPECIALIZADA.': 9,
    'Emissão Bilhete Aéreo': 10,
    'HOSPEDAGEM ,EXCETO DO PARLAMENTAR NO DISTRITO FEDERAL.': 11,
    'SERVIÇO DE TÁXI, PEDÁGIO E ESTACIONAMENTO': 12,
    'PARTICIPAÇÃO EM CURSO, PALESTRA OU EVENTO SIMILAR': 13,
    'LOCAÇÃO OU FRETAMENTO DE AERONAVES': 14,
    'PASSAGENS TERRESTRES, MARÍTIMAS OU FLUVIAIS': 15,
    'ASSINATURA DE PUBLICAÇÕES': 16,
    'LOCAÇÃO DE VEÍCULOS AUTOMOTORES OU FRETAMENTO DE EMBARCAÇÕES': 17,
    'LOCAÇÃO OU FRETAMENTO DE EMBARCAÇÕES': 18,
    'LOCOMOÇÃO, ALIMENTAÇÃO E  HOSPEDAGEM': 19,
    'AQUISIÇÃO DE MATERIAL DE ESCRITÓRIO.': 20,
    'AQUISIÇÃO OU LOC. DE SOFTWARE; SERV. POSTAIS; ASS.': 21
}

EXCLUIR_PARTIDOS = ['LIDERANÇA DO PSDB', 'LIDERANÇA DO PT', 'LIDMIN', 'NOVO',
                    'PDT', 'PODE', 'PP', 'PROS', 'PSD', 'PTB', 'SDD']