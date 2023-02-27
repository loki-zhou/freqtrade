import openai

# 设置API密钥
openai.api_key = "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"

# 聊天机器人要回复的信息
prompt = "甲对乙说，如果你给我一本书，我的书就和你一样多。乙对甲说，如果你给我一本书，我就是你的两倍。甲乙各有多少本书？"

# 发送API请求
response = openai.Completion.create(
    engine="davinci-codex",
    prompt=prompt,
    max_tokens=1200,
    temperature=0.7,
)

# 获取生成的文本
completions = response.choices
for completion in completions:
    print(completion.text)