def generate_prompt(data_point):
    """
    Generate a formatted prompt for fine-tuning from a data_point
    :param data_point: a dictionary containing the context and response
    :return: a string containing the formatted prompt
    """
    prompt_parts = list()
    for i in range(0, len(data_point), 2):
        context = data_point[i].split(':', 1)[-1].strip()
        prompt_parts.append(f'<INST> {context} </INST>')

        try:
            response = data_point[i + 1].split(':', 1)[-1].strip()
            prompt_parts.append(f'{response}')
        except:
            pass

    prompt = ' '.join(prompt_parts) + ' </s>'

    return prompt.strip()


def generate_prompt_in_batch(data_point):
    contexts = data_point['utterances']
    prompts = []

    for data in contexts:
        prompt = generate_prompt(data)
        prompts.append(prompt)

    return prompts
