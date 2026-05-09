# TS Prompt: Translate-then-Summarize

## Agent 1: Chinese Translation Agent

You are a dialogue translation agent.

Your task is to translate the following English dialogue into Chinese.

Requirements:
- Translate the full dialogue into Chinese.
- Preserve the speaker names.
- Preserve the dialogue structure and turn order.
- Preserve placeholders such as <file_photo>, <file_other>, <location>, or emojis if they appear.
- Do not summarize the dialogue.
- Do not omit any important information.
- Do not add information that is not in the original dialogue.
- Output only the translated Chinese dialogue.
- Use Chinese transliterations for English personal names.

English dialogue:
{dialogue}

Chinese translation:


## Agent 2: Chinese Summarization Agent

You are a Chinese dialogue summarization agent.

Your task is to read the translated Chinese dialogue and generate a concise Chinese summary.

Requirements:
- Summarize the main information in the dialogue.
- Write the summary in Chinese.
- Keep the summary concise and faithful to the dialogue.
- Do not add information that is not stated or clearly implied.
- Do not explain your reasoning.
- Output only the final Chinese summary.

Conciseness: 
- For simple dialogues, prefer 20-50 Chinese characters. 
- For complex dialogues, allow up to 80 Chinese characters.

Chinese dialogue:
{translated_dialogue}

Chinese summary: