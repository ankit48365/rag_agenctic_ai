from model import opus_response, sonet_response, haiku_response

def call_all_models(system_prompt, user_prompt):
    opus_result = opus_response(system_prompt, user_prompt)
    # sonnet_result = sonet_response(system_prompt, user_prompt)
    haiku_result = haiku_response(system_prompt, user_prompt)
    print("--------------------------------------------------------------------")
    print("Opus Response:\n", opus_result.content)
    # print("\nGranite Response:\n", sonnet_result.content)
    print("--------------------------------------------------------------------")
    print("\nHaiku Response:\n", haiku_result.content)
    print("--------------------------------------------------------------------")


# Example call to test all models
call_all_models("You are a helpful assistant who provides concise and accurate answers", "What is the capital of Canada? Tell me a cool fact about it as well")