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


API_PRICING={
    MODEL_GPT3: 0.002,
    MODEL_GPT4: 0.03,
    MODEL_DAVINCI3: 0.02
}

API_KEYS_FILE="../api_keys_20230324.json"
API_KEYS_FILE_2="../api_keys_20230612.json"

SYSTEM_ROLE = {
    'role': "system", 
    'content': "Você avalia se uma passagem de texto responde a uma pergunta, indicando uma pontuação de 0 à 10, onde 0 indica que a passagem não responde e 10 que a passagem responde de forma correta e clara. Você desconsidera informações que o texto diz que vai apresentar mas não apresenta."
    # 'content': "Você avalia se uma passagem de texto responde a uma pergunta, indicando uma pontuação de 0 à 10, onde 0 indica que a passagem não responde e 10 que a passagem responde diretamente. Você desconsidera informações que o texto diz que vai apresentar mas não apresenta."
}

FEW_SHOT_EXAMPLES=[
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


SINGLE_FEW_SHOTS_PROMPT="Você avalia se uma passagem de texto responde a uma pergunta, indicando uma pontuação de 0 à 10, onde 0 indica que a passagem não responde e 10 que a passagem responde diretamente. Você desconsidera informações que o texto diz que vai apresentar mas não apresenta. Siga os exemplos abaixo.\n\nExemplo 1:\nPassagem: \"O Brasil possui muitas belezas naturais. Neste artigo vamos indicar os melhores lugares para passear no Brasil.\"\nPergunta: \"Onde passear no Brasil?\"\n\nPontuação: 1; Razão: a passagem apenas indica que o Brasil tem muitas belezas naturais, mas não indica nenhum exemplo, embora a passagem indique que  artigo vai falar sobre lugares para passear no Brasil, mas o trecho apresentado não lista nenhum lugar específico para passear no Brasil.\n\nExemplo 2:\nPassagem: \"O cirurgião faz uma incisão no quadril, remove a articulação do quadril danificada e a substitui por uma articulação artificial que é uma liga metálica ou, em alguns casos, cerâmica. A cirurgia geralmente leva cerca de 60 a 90 minutos para ser concluída.\"\n\nPergunta: \"de que metal são feitas as próteses de quadril?\"\n\nPontuação: 2; Razão: não responde a pergunta de forma clara, pois apenas indica indiretamente que a prótese pode ser de uma liga metálica, mas não explicita quais metais. O assunto da passagem é sobre cirurgia de colocação de prótese que, embora relacionado, não é diretamente o assunto da pergunta.\n\n"


ADDITIONAL_FEW_SHOT_EXAMPLES=[
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

FEW_SHOT_EXAMPLE_FORMAT="Exemplo {}:\n{}"

SINGLE_FEW_SHOT_PROMPT_INITIAL_FORMAT="{} Siga os exemplos abaixo."

QUERY_PASSAGE_FORMAT="Passagem: \"{}\"\nPergunta: \"{}\""
OPENAI_RESPONSE_REGEX="[\n\r]*[Pp]ontuação:\s*(.+)\s*;\s*[Rr]azão:\s*(.+)[\n\r]*"

MAX_TOKENS_RESPONSE=500



def initialize_openai(which_key="OPENAI_API_KEY"):

    if which_key == "OPENAI_API_KEY_2":
        api_keys_filename = API_KEYS_FILE_2
    else:
        api_keys_filename = API_KEYS_FILE

    with open(api_keys_filename) as inputFile:
        api_keys = json.load(inputFile)

    openai.api_key = api_keys[which_key]



def execute_LLM_passage_relevance_evaluation(which_query, 
                                             which_passage, 
                                             model=MODEL_GPT3, 
                                             verbose=True):

    start_time = time.time()

    query_passage_to_evaluate=QUERY_PASSAGE_FORMAT.format(which_passage, which_query)

    if verbose:
        print("++++++++++++++++++++++++++")
        print(query_passage_to_evaluate)
        print("++++++++++++++++++++++++++")

    if model in [MODEL_GPT3, MODEL_GPT4]:
        messages_to_send = [SYSTEM_ROLE]

        for i, example in enumerate(FEW_SHOT_EXAMPLES):
            for example_role in example:
                if example_role['role'] == "user":
                    messages_to_send.append({'role': "user", 
                                             'content': FEW_SHOT_EXAMPLE_FORMAT.format(i + 1, example_role['content'])})
                else:
                    messages_to_send.append(example_role)

        messages_to_send.append({'role': "user", 'content': query_passage_to_evaluate})

    
        if verbose:
            print("\n")
            print(messages_to_send)
        
        response = openai.ChatCompletion.create(model=model,
                                                messages=messages_to_send,
                                                temperature=0,
                                                max_tokens=MAX_TOKENS_RESPONSE)
        
        response_text = response['choices'][0]['message']['content']

    elif model == MODEL_DAVINCI3:
        prompt_to_send = SINGLE_FEW_SHOT_PROMPT_INITIAL_FORMAT.format(SYSTEM_ROLE['content'])

        # for i, example in enumerate([FEW_SHOT_EXAMPLES[0], FEW_SHOT_EXAMPLES[1] ,FEW_SHOT_EXAMPLES[2], ADDITIONAL_FEW_SHOT_EXAMPLES[0]]):
        for i, example in enumerate([FEW_SHOT_EXAMPLES[0], FEW_SHOT_EXAMPLES[1] ,FEW_SHOT_EXAMPLES[2]]):
            for example_role in example:
                if example_role['role'] == "user":
                    prompt_to_send += "\n\n" + FEW_SHOT_EXAMPLE_FORMAT.format(i + 1, example_role['content'])
                else:
                    prompt_to_send += "\n\n" + example_role['content']

        prompt_to_send += "\n\n" + query_passage_to_evaluate

        # prompt_to_send = SINGLE_FEW_SHOTS_PROMPT + query_passage_to_evaluate

        if verbose:
            print("\n")
            print(prompt_to_send)


        response = openai.Completion.create(model=model,
                                            prompt=prompt_to_send,
                                            temperature=0,
                                            max_tokens=MAX_TOKENS_RESPONSE,
                                            top_p=1,
                                            frequency_penalty=0,
                                            presence_penalty=0)  
        
        response_text = response['choices'][0]['text']

    else:
        raise ValueError("Cannot handle OPENAI model {}...".format(model))


    if verbose:
        print("\n")
        print(response_text)

    m = re.match(OPENAI_RESPONSE_REGEX, response_text)

    if len(m.groups()) == 2:
        score = int(m.group(1))
        reasoning = m.group(2)
    else:
        score = None
        reasoning = None

    final_time = time.time()

    print("\nLLM document aggregation duration: {}\n\n".format(final_time - start_time))

    return {'score': score,
            'reasoning': reasoning,
            'usage': response['usage'].copy(),
            'cost': response['usage']['total_tokens'] / 1000 * API_PRICING[model],
            'duration': final_time - start_time}
