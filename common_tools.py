import pandas as pd
import json
import openai
import time

import re

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 200)


MODEL_GPT3 = "gpt-3.5-turbo"
MODEL_GPT4 = "gpt-4"
MODEL_DAVINCI3 = "text-davinci-003"


API_PRICING_INPUT={
    MODEL_GPT3: 0.0015,
    MODEL_GPT4: 0.03,
    MODEL_DAVINCI3: 0.02
}

API_PRICING_OUTPUT={
    MODEL_GPT3: 0.002,
    MODEL_GPT4: 0.06,
    MODEL_DAVINCI3: 0.02
}

API_KEYS_FILE="../api_keys_20230324.json"
API_KEYS_FILE_2="../api_keys_20230612.json"


#
# Queries LLM evaluation definitions
#

EVALUATION_SYSTEM_ROLE = {
    'role': "system", 
    'content': "Você avalia se uma passagem de texto responde a uma pergunta, indicando uma pontuação de 0 à 10, onde 0 indica que a passagem não responde e 10 que a passagem responde de forma correta e clara. Você desconsidera informações que o texto diz que vai apresentar mas não apresenta."
}

EVALUATION_FEW_SHOT_EXAMPLES=[
    [
        {
            'role': "user",
            'content': "Passagem: \"O cirurgião faz uma incisão no quadril, remove a articulação do quadril danificada e a substitui por uma articulação artificial que é uma liga metálica ou, em alguns casos, cerâmica. A cirurgia geralmente leva cerca de 60 a 90 minutos para ser concluída.\"\nPergunta: \"de que metal são feitas as próteses de quadril?\"", 
        },
        {  
            'role': "assistant",
            'content': "Pontuação: 2; Razão: não responde a pergunta de forma clara, pois apenas indica indiretamente que a prótese pode ser de uma liga metálica, mas não explicita quais metais. O assunto da passagem é sobre cirurgia de colocação de prótese que, embora relacionado, não é diretamente o assunto da pergunta."
        }
    ],

    [
        {
            'role': "user",
            'content': "Passagem: \"Vanessa Redgrave nasceu em 30 de janeiro de 1937 em Londres. Ela fará falta, mas não será esquecida. Centenas de fãs imediatamente começaram a escrever suas mensagens de condolências na página do Facebook, expressando sua tristeza pela morte da talentosa atriz de 79 anos.\"\nPergunta: \"quantos anos tem vanessa redgrave?\"",
        },
        {
            'role': "assistant",
            'content': "Pontuação: 8; Razão: a passagem responde a idade que Vanessa Redgrave tinha quando faleceu em 2016, informação que exige que o leitor faça o cálculo entre a idade da atriz e o seu ano de nascimento."
        }
    ],

    [
        {
            'role': "user",
            'content': "Passagem: \"O Brasil possui muitas belezas naturais. Neste artigo vamos indicar os melhores lugares para passear no Brasil.\"\nPergunta: \"Onde passear no Brasil?\"",
        },
        {
            'role': "assistant",
            'content': "Pontuação: 1; Razão: a passagem apenas indica que o Brasil tem muitas belezas naturais, mas não indica nenhum exemplo. Embora a passagem indique que artigo vai falar sobre lugares para passear no Brasil, o trecho apresentado não lista nenhum lugar específico para passear no Brasil."
        }
    ]    
]

EVALUATION_ADDITIONAL_FEW_SHOT_EXAMPLES=[
    [
        {
            'role': "user",
            'content': "Passagem: \"Uma prótese feita de metal e plástico são os implantes de substituição do quadril mais comumente usados. Tanto a bola quanto o soquete da articulação do quadril são substituídos por um implante de metal e um espaçador de plástico é colocado entre eles. Os metais mais comumente usados incluem titânio e aço inoxidável.\"\nPergunta: \"de que metal são feitas as próteses de quadril?\"",
        },
        {
            'role': "assistant",
            'content': "Pontuação: 10; Razão: a passagem responde à pergunta de forma clara e direta, e ainda acrescenta informações relevantes sobre próteses de quadril."
        }
    ],

    [
        {
            'role': "user",
            'content': "Passagem: \"conveniente no bairro mais moderno da cidade, o Radisson Blu Belo Horizonte, Savassi é um dos hotéis cinco estrelas mais desejados do Sudeste do Brasil. Passe o dia comprando produtos de marcas famosas no Pátio Savassi ou visitando a Pampulha, um Patrimônio Mundial da UNESCO maravilhosamente preservado, antes de voltar para um dos nossos quartos de hotel modernos. Quando estiver pronto para se aventurar novamente, caminhe pelo movimentado Mercado Central de Belo Horizonte para comprar souvenires ou frutas frescas e provar lanches tradicionais brasileiros, como o pão de queijo. Comece seu dia com o pé direito saboreando um café da manhã de cortesia na Pizzaria Olegário e volte mais tarde para um delicioso almoço ou jantar. Viajando a trabalho? Com um centro de negócios, estacionamento amplo e quatro espaços de\"\nPergunta: \"Onde passear no Brasil?\"",
        },
        {
            'role': "assistant",
            'content': "Pontuação: 5; Razão: a passagem responde à pergunta muito indiretamente, indicando que na região do hotel em Belo Horizonte fica um Patrimônio Mundial da UNESCO, que pode ser uma boa sugestão de passeio no Brasil."
        }
    ],
    
    [
        {
            'role': "user",
            'content': "Passagem: \"ideal para a sua viagem à bela Cidade Jardim do Brasil. Quando suas reuniões do dia terminarem, você pode caminhar até o restaurante Dona Lucinha para saborear a culinária local e, depois, visitar as lojas no Pátio Savassi. Se preferir passar algum tempo admirando a paisagem exuberante da área, há praças públicas bem cuidadas, como a Praça da Liberdade e a Praça da Savassi a uma curta distância a pé. Parque Municipal Américo Renné Giannetti 1,23 mi / 1,97 km do hotel Deixe-se envolver pela natureza neste belo parque no centro da cidade de Belo Horizonte. Desfrute de um piquenique na grama, alugue um barco a remo para passear no lago ou observe seus filhos gastarem sua energia no playground. Praça da Savassi 0,23 mi / 0,3\"\nPergunta: \"Onde passear no Brasil?\"",
        },
        {
            'role': "assistant",
            'content': "Pontuação: 1; Razão: a passagem não responde à pergunta, indicando detalhes de uma região de Belo Horizonte que dificilmente vão interessar alguém buscando informações gerais sobre passeios no Brasil."
        }
    ]    
]

EVALUATION_FEW_SHOT_EXAMPLE_FORMAT="Exemplo {}:\n{}"

EVALUATION_SINGLE_FEW_SHOT_PROMPT_INITIAL_FORMAT="{} Siga os exemplos abaixo."

EVALUATION_QUERY_PASSAGE_FORMAT="Passagem: \"{}\"\nPergunta: \"{}\""
EVALUATION_OPENAI_RESPONSE_REGEX="[\n\r]*[Pp]ontuação:\s*(.+)\s*;\s*[Rr]azão:\s*(.+)[\n\r]*"

EVALUATION_MAX_TOKENS_RESPONSE=500

#
# Queries LLM creation definitions
#

CREATION_SYSTEM_ROLE = {
    'role': "system", 
    'content': "Você sugere 2 perguntas a partir da leitura de uma passagem de texto. A primeira pergunta explora o tema da passagem, e a segunda pergunta explora uma informação ou conclusão específica possível a partir da leitura da passagem. Suas perguntas devem fazer sentido para alguém que não leu a passagem. Siga o formato do exemplo."
}

CREATION_ONE_SHOT_EXAMPLE=[
    {
        'role': "user",
        'content': "Exemplo: Passagem: \"Como acontece com todos os tratamentos naturais, a qualidade do produto utilizado para o tratamento de decide o resultado. Portanto, se você deseja obter os melhores resultados com o tratamento óleo de rosa mosqueta para a acne, você deve tentar encontrar o melhor e mais puro óleo de rosa mosqueta orgânica. Antes de comprar um produto, certifique-se que você leia os rótulos das embalagens adequadamente para verificar se ele contém óleo de rosa mosqueta puro ou de uma mistura de outros óleos essenciais. Leia as instruções de uso recomendadas pelo fabricante, porque alguns produtos requerem lavagem após alguns minutos da aplicação, enquanto que alguns precisam ser mantidos durante a noite.óleo de rosa mosqueta tem um cheiro desagradável e desagradável e muitas pessoas podem não gostar. Se você tem crianças em casa, eles podem ser desligados de você devido ao cheiro. Por isso, certifique-se de que você adicionar uma certa quantidade de óleo essencial aromático, tal como lavanda ou jasmim para travar para baixo o cheiro.[ Ler: Como usar o óleo de abacate para acne? ]Considerações ao usar o Óleo de Rosa Mosqueta\"", 
    },
    {  
        'role': "assistant",
        'content': "Pergunta 1: Quais tratamentos para acne?\nPergunta 2: Como é possível evitar o forte cheiro da rosa mosqueta no tratamento de pele?"
    }
]

CREATION_PASSAGE_FORMAT="Passagem: \"{}\""
CREATION_OPENAI_RESPONSE_REGEX="[\n\r]*[Pp]ergunta 1:\s*(.+)\s*[\n\r]+[Pp]ergunta 2:\s*(.+)[\n\r]*"

CREATION_MAX_TOKENS_RESPONSE=200





def initialize_openai(which_key="OPENAI_API_KEY"):

    if which_key == "OPENAI_API_KEY_2":
        api_keys_filename = API_KEYS_FILE_2
    else:
        api_keys_filename = API_KEYS_FILE

    with open(api_keys_filename) as inputFile:
        api_keys = json.load(inputFile)

    openai.api_key = api_keys[which_key]



def compute_openai_api_usage_cost(api_usage_dict, which_model):

    cost =  api_usage_dict['prompt_tokens'] / 1000 * API_PRICING_INPUT[which_model] + \
            api_usage_dict['completion_tokens'] / 1000 * API_PRICING_OUTPUT[which_model]
    
    return cost



def execute_LLM_passage_relevance_evaluation(which_query, 
                                             which_passage, 
                                             model=MODEL_GPT3, 
                                             verbose=True):

    start_time = time.time()

    query_passage_to_evaluate=EVALUATION_QUERY_PASSAGE_FORMAT.format(which_passage, which_query)

    if verbose:
        print("++++++++++++++++++++++++++")
        print(query_passage_to_evaluate)
        print("++++++++++++++++++++++++++")

    if model == MODEL_GPT3:
        messages_to_send = [EVALUATION_SYSTEM_ROLE]

        for i, example in enumerate(EVALUATION_FEW_SHOT_EXAMPLES):
            for example_role in example:
                if example_role['role'] == "user":
                    messages_to_send.append({'role': "user", 
                                             'content': EVALUATION_FEW_SHOT_EXAMPLE_FORMAT.format(i + 1, example_role['content'])})
                else:
                    messages_to_send.append(example_role)

        messages_to_send.append({'role': "user", 'content': query_passage_to_evaluate})

    
        if verbose:
            print("\n")
            print(messages_to_send)
        
        response = openai.ChatCompletion.create(model=model,
                                                messages=messages_to_send,
                                                temperature=0,
                                                max_tokens=EVALUATION_MAX_TOKENS_RESPONSE)
        
        response_text = response['choices'][0]['message']['content']

    elif model == MODEL_GPT4:
        messages_to_send = [EVALUATION_SYSTEM_ROLE]

        for i, example in enumerate([EVALUATION_FEW_SHOT_EXAMPLES[0], EVALUATION_FEW_SHOT_EXAMPLES[2]]):
            for example_role in example:
                if example_role['role'] == "user":
                    messages_to_send.append({'role': "user", 
                                             'content': EVALUATION_FEW_SHOT_EXAMPLE_FORMAT.format(i + 1, example_role['content'])})
                else:
                    messages_to_send.append(example_role)

        messages_to_send.append({'role': "user", 'content': query_passage_to_evaluate})

    
        if verbose:
            print("\n")
            print(messages_to_send)
        
        response = openai.ChatCompletion.create(model=model,
                                                messages=messages_to_send,
                                                temperature=0,
                                                max_tokens=EVALUATION_MAX_TOKENS_RESPONSE)
        
        response_text = response['choices'][0]['message']['content']

    elif model == MODEL_DAVINCI3:
        prompt_to_send = EVALUATION_SINGLE_FEW_SHOT_PROMPT_INITIAL_FORMAT.format(EVALUATION_SYSTEM_ROLE['content'])

        for i, example in enumerate([EVALUATION_FEW_SHOT_EXAMPLES[0], EVALUATION_FEW_SHOT_EXAMPLES[1] ,EVALUATION_FEW_SHOT_EXAMPLES[2]]):
            for example_role in example:
                if example_role['role'] == "user":
                    prompt_to_send += "\n\n" + EVALUATION_FEW_SHOT_EXAMPLE_FORMAT.format(i + 1, example_role['content'])
                else:
                    prompt_to_send += "\n\n" + example_role['content']

        prompt_to_send += "\n\n" + query_passage_to_evaluate

        if verbose:
            print("\n")
            print(prompt_to_send)


        response = openai.Completion.create(model=model,
                                            prompt=prompt_to_send,
                                            temperature=0,
                                            max_tokens=EVALUATION_MAX_TOKENS_RESPONSE,
                                            top_p=1,
                                            frequency_penalty=0,
                                            presence_penalty=0)  
        
        response_text = response['choices'][0]['text']

    else:
        raise ValueError("Cannot handle OPENAI model {}...".format(model))


    if verbose:
        print("\n")
        print(response_text)

    m = re.match(EVALUATION_OPENAI_RESPONSE_REGEX, response_text)

    if len(m.groups()) == 2:
        score = int(m.group(1))
        reasoning = m.group(2)
    else:
        score = None
        reasoning = None

    final_time = time.time()
    final_cost = compute_openai_api_usage_cost(response['usage'], model)

    print("\nLLM query relevance evaluation duration: {}; cost: {}\n\n".format(final_time - start_time, final_cost))

    return {'score': score,
            'reasoning': reasoning,
            'usage': response['usage'].copy(),
            'cost': final_cost,
            'duration': final_time - start_time}



def execute_LLM_query_creation(which_passage, 
                               model=MODEL_GPT3, 
                               verbose=True):

    start_time = time.time()

    passage_to_create_question=CREATION_PASSAGE_FORMAT.format(which_passage)

    if verbose:
        print("++++++++++++++++++++++++++")
        print(passage_to_create_question)
        print("++++++++++++++++++++++++++")

    if model == MODEL_GPT3:
        messages_to_send = [CREATION_SYSTEM_ROLE]

        for example_role in CREATION_ONE_SHOT_EXAMPLE:
            if example_role['role'] == "user":
                messages_to_send.append({'role': "user", 
                                         'content': example_role['content']})
            else:
                messages_to_send.append(example_role)

        messages_to_send.append({'role': "user", 'content': passage_to_create_question})

    
        if verbose:
            print("\n")
            print(messages_to_send)
        
        response = openai.ChatCompletion.create(model=model,
                                                messages=messages_to_send,
                                                temperature=0,
                                                max_tokens=CREATION_MAX_TOKENS_RESPONSE)
        
        response_text = response['choices'][0]['message']['content']

    else:
        raise ValueError("Cannot handle OPENAI model {}...".format(model))


    if verbose:
        print("\n")
        print(response_text)

    m = re.match(CREATION_OPENAI_RESPONSE_REGEX, response_text)

    if (m is not None) and (len(m.groups()) == 2):
        question_theme = m.group(1)
        question_specific = m.group(2)
    else:
        question_theme = None
        question_specific = None

    final_time = time.time()

    final_cost = compute_openai_api_usage_cost(response['usage'], model)

    print("\nLLM passage queries creation duration: {}; cost: {}\n\n".format(final_time - start_time, final_cost))

    return {'question_theme': question_theme,
            'question_specific': question_specific,
            'usage': response['usage'].copy(),
            'cost': final_cost,
            'duration': final_time - start_time}
