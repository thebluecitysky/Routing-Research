from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class DeepSeekModel:
    def __init__(
        self,
        model_path="deepseek/DeepSeek-R1-Distill-Qwen-1.5B",
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # 加载tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

        # 设置模型为评估模式
        self.model.eval()

    def generate_response(self, prompt, max_length=2048, temperature=0.7):
        # try:
        # 对输入进行编码
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # 生成回答
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # 解码输出
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

        # except Exception as e:
        #     print(f"生成回答时发生错误: {str(e)}")
        #     return None


"7:10-8:00：狗狗在客厅活动。8:00-8:30：狗狗在吃饭。8:30-11:30：狗狗在花园内自由活动，并进行了便便。11:30-13:40：狗狗在客厅的窝中睡觉。14:00-15:00：狗狗多次上餐桌，最长一次超过20分钟，期间没有人经过。15:00-17:00：狗狗在沙发上休息。"


def main():
    # 初始化模型
    model = DeepSeekModel()

    # 测试对话
    # while True:
    #     user_input = input("\n请输入您的问题 (输入 'q' 退出): ")
    #     if user_input.lower() == "q":
    #         break

    #     response = model.generate_response(user_input)
    #     if response:
    #         print("\nDeepSeek:", response)

    chat_history = [
        {
            "role": "system",
            "content": "你是一个ai看家助手,你需要根据我给你的家庭成员时间活动分析问题,50字以内简要回答。",
        }
    ]
    while True:
        user_input = input("user:")
        if user_input.lower() in ["exit"]:
            break
        chat_history.append({"role": "user", "content": user_input})
        text = model.tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=True
        )
        chat_prompt = model.tokenizer([text], return_tensors="pt").to(model.device)
        output = model.model.generate(
            **chat_prompt,
            max_new_tokens=512,
            pad_token_id=model.tokenizer.eos_token_id,
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(chat_prompt.input_ids, output)
        ]
        response = model.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        print(f"\nanswer：{response}")
        chat_history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
