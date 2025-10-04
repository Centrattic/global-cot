from openai_harmony import (
    load_harmony_encoding,
    HarmonyEncodingName,
)

test_gpt_oss = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
print(test_gpt_oss)
