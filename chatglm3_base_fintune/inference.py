import argparse
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
import os
import json
from peft import get_peft_model, LoraConfig, TaskType
from preprocess_utils import sanity_check, InputOutputDataset

# Argument Parser Setup
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="/mnt/haofei/MSA/ChatGLM3/chatglm3-6b-base", help="The directory of the model")
parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer path")
parser.add_argument("--LoRA", type=str, default=True, help="use lora or not")
parser.add_argument("--erc_lora_path", type=str,default='chatglm3_base_fintune/ChatGLM3/ChatGLM_LoRA/ERC_lora/pytorch_model.pt',
                    help="Path to the ERC LoRA model checkpoint")
parser.add_argument("--ecp_lora_path", type=str, default='chatglm3_base_fintune/ChatGLM3/ChatGLM_LoRA/ECP_lora/pytorch_model.pt', 
                    help="Path to the ECP LoRA model checkpoint")
parser.add_argument("--device", type=str, default="cuda", help="Device to use for computation")
parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum new tokens for generation")
parser.add_argument("--lora-alpha", type=float, default=32, help="LoRA alpha")
parser.add_argument("--lora-rank", type=int, default=8, help="LoRA r")
parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout")
parser.add_argument("--gold_path", type=str, default='chatglm3_base_fintune/data/Subtask_2_test.json', help="test data path")
parser.add_argument("--erc_test_path", type=str, default='chatglm3_base_fintune/data/subtask2_test_ERC.json', help="erc json path")
parser.add_argument("--ecp_test_path", type=str, default='chatglm3_base_fintune/data/subtask2_test_ECP.json', help="ecp json path")
parser.add_argument("--gold_emo_path", type=str, default='chatglm3_base_fintune/pred_results/Subtask_2_emo_pred.json', help="erc pred save path")
parser.add_argument("--save_pred_path", type=str, default='chatglm3_base_fintune/pred_results/Subtask_2_pred.json', help="result save path")

args = parser.parse_args()

def get_response(args, lora_path, test_data_path):
    if args.tokenizer is None:
        args.tokenizer = args.model
    if args.LoRA:
        # Model and Tokenizer Configuration
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        model = AutoModel.from_pretrained(args.model, load_in_8bit=False, trust_remote_code=True, device_map="auto").to(
            args.device)
        # LoRA Model Configuration
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=True,
            target_modules=['query_key_value'],
            r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout
        )
        model = get_peft_model(model, peft_config)
        if os.path.exists(lora_path):
            model.load_state_dict(torch.load(lora_path), strict=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        model = AutoModel.from_pretrained(args.model, load_in_8bit=False, trust_remote_code=True, device_map="auto").to(
            args.device)

    # inference
    response_list = []
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    for data in test_data:
        prompt = data['context']
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        response = model.generate(input_ids=inputs["input_ids"],
                                  max_length=inputs["input_ids"].shape[-1] + args.max_new_tokens)
        response = response[0, inputs["input_ids"].shape[-1]:]
        decode_response = tokenizer.decode(response, skip_special_tokens=True)
        response_list.append(decode_response)
        print("Response:", decode_response)
    return response_list

def erc_pred(args):
    erc_test_path = args.erc_test_path
    lora_path = args.erc_lora_path
    if os.path.exists(args.ecp_test_path) and os.path.exists(args.gold_emo_path):
        print("ECP test JSON file already exists, use the existing file!")
    else:
        response_list = get_response(args, lora_path, erc_test_path)
        print("Emotion pred done, building ECP task \n")
        # formalized prediction results
        with open(args.gold_path, "r", encoding="utf-8") as f:
            gold_data = json.load(f)
        num = 0
        for i, gold in enumerate(gold_data):
            for j in range(len(gold['conversation'])):
                emotion_labels = response_list[num + j]
                gold_data[i]['conversation'][j]["emotion"] = emotion_labels
            num += j + 1
            # save as a json file
        json_data = json.dumps(gold_data)
        with open(args.gold_emo_path, "w") as f:
            f.write(json_data)
        print("ERC pred results has saved")
        build_ecp_task(gold_data)

def ecp_pred(args):
    ecp_test_path = args.ecp_test_path
    lora_path = args.ecp_lora_path
    response_list = get_response(args, lora_path, ecp_test_path)
    emo_list = ['neutral', 'joy', 'surprise', 'sadness', 'disgust', 'anger', 'fear']
    # formalized prediction results
    with open(args.gold_emo_path, "r", encoding="utf-8") as f:
        gold_data = json.load(f)
        num = 0
        for i, gold in enumerate(gold_data):
            triple = []
            for j in range(len(gold['conversation'])):
                emotion = gold['conversation'][j]['emotion']
                if emotion not in emo_list:
                    emotion = 'neutral'
                cause_indices = response_list[num + j]
                cause_list = [f'{item}' for item in cause_indices.split(',')][:-1]
                for cause_index in cause_list:
                    if cause_index != '0' and cause_index != '':
                        triple.append([str(j + 1) + '_' + emotion, str(cause_index)])
                gold_data[i]["emotion-cause_pairs"] = triple
            num += j+1
    # save as a json file
    json_data = json.dumps(gold_data)
    with open(args.save_pred_path, "w") as f:
        f.write(json_data)

def build_ecp_task(gold_data):
    prefix = (
    "You're an expert in sentiment analysis and emotion cause identification. I'm going to give you a conversation "
    "raised by several speakers and its corresponding emotion label. Please inference the emotion-cause indices of the utterances from the given candidate "
    "utterances that cause the emotion in the target utterance."
    "Here's a demonstrate: "
    "{"
    "input conversation: "
    "1_joy.Chandler:Hey Pheebs! "
    "2_surprise.Phoebe:Ohh!You made up! "
    "3_joy.Monica:Yeah,I couldn't be mad at him for too long. "
    "4_joy.Chandler:Yeah,she couldn't live without the Chan Love. "
    "5_disgust.Phoebe:Ohh,get a room. "
    "candidate utterances:"
    "1_joy.Chandler:Hey Pheebs! "
    "2_surprise.Phoebe:Ohh!You made up! "
    "3_joy.Monica:Yeah,I couldn't be mad at him for too long. "
    "target utterance:"
    "4_joy.Chandler: Yeah, she couldn't live without the Chan Love."
    "The emotion-cause indices of the target utterance is: "
    "2 "
    "}."
    )
    conversations = []
    for num, lines in enumerate(gold_data):
        sour_lines = gold_data[num]
        utterances = ""
        utterance_list = []
        emotions = []
        for i, utt in enumerate(lines['conversation']):
            text = utt['text']
            speaker = utt['speaker']
            emotion = sour_lines['conversation'][i]['emotion']
            utterance = f'{i + 1}' + '_' + emotion + '. ' + speaker + ':' + text + ' '
            utterances += utterance
            utterance_list.append(utterance)
            emotions.append(emotion)
        utterance = ''
        for i, utt in enumerate(utterance_list):
            utterance += utt
            context = ('{0}input conversation: {1}candidate utterances: {2} target utterance: {3} The emotion-cause '
                    'indices of the target utterance are: '.format(prefix, utterances, utterance, utt))
            conversation = {'context': context}
            conversations.append(conversation)
    json_data = json.dumps(conversations)
    with open(args.ecp_test_path, "w") as f:
        f.write(json_data)
    print("Successfully bulid ECP task! ")

def build_erc_task(args):
    prefix = ("You're an expert in sentiment analysis. I'm going to give you a conversation with multiple "
          "utterances raised by certain speakers. Your task is to comprehensively and thoroughly analyze the entire conversation"
          "and then determine the most suitable emotion labels from {neutral, joy, surprise, sadness, disgust, anger, fear}"
          "for the target utterance."
          "Here's an example: "
          "{"
          "input conversation: "
          "1.Chandler:Hey Pheebs! "
          "2.Phoebe:Ohh!You made up! "
          "3.Monica:Yeah,I couldn't be mad at him for too long. "
          "4.Chandler:Yeah,she couldn't live without the Chan Love. "
          "5.Phoebe:Ohh,get a room. "
          "target utterance:"
          "5.Phoebe:Ohh,get a room."
          "The emotion labels of the target utterance is: "
          "disgust "
          "}."
          )

    with open(args.gold_path, 'r') as f:
        data = json.load(f)

    num_conversation = 0
    conversations = []
    for lines in data:
        num_conversation += 1
        utterances = ""
        utterance_list = []
        for i, utt in enumerate(lines['conversation']):
            text = utt['text']
            speaker = utt['speaker']
            utterance = f'{i + 1}:' + speaker + ':' + text + ' '
            utterances += utterance
            utterance_list.append(utterance)
        utterance = ''
        for i, utt in enumerate(utterance_list):
            utterance += utt
            context = ('{0}input conversation: {1} target utterance: {2} ''The emotion label of the target utterance are: '
                    .format(prefix, utterances, utt))

            conversation = {'context': context}
            conversations.append(conversation)
    json_data= json.dumps(conversations)
    with open(args.erc_test_path, "w") as f:
        f.write(json_data)
    print("Successfully bulid ERC task! ")

if __name__ == '__main__':
    if not os.path.exists(args.erc_test_path):
        build_erc_task(args)
    else:
        print("ERC test JSON file already exists, use the existing file!")
    erc_pred(args)
    ecp_pred(args)