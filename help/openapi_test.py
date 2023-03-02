import openai

# 设置API密钥
openai.api_key = "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"

# 聊天机器人要回复的信息
prompt = "甲对乙说，如果你给我一本书，我的书就和你手里一样多，乙对甲说，如果你给我一本书，我就是你手里的两倍。甲乙各有多少本书?"

# 发送API请求
# response = openai.Completion.create(
#     engine="davinci-codex",
#     prompt=prompt,
#     max_tokens=1200,
#     temperature=0.7,
# )
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
)

# 获取生成的文本
# completions = response.choices
# for completion in completions:
#     print(completion.message)
#print(response.choices)
for item in response.choices:
    print(item.message.content)