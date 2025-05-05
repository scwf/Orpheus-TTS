from orpheus_tts import OrpheusModel
import wave
import time
import os
import torch
from pathlib import Path

# 这个方法是错误的，不要使用
def main():
    try:
        # 配置参数
        model_name = "canopylabs/3b-zh-pretrain-research_release"
        output_file = "output-cloned.wav"
        temperature = 0.6
        top_p = 0.8
        repetition_penalty = 1.2
        
        # 获取qjc_sample_data目录的绝对路径
        data_dir = os.path.join(os.getcwd(), "qjc_sample_data")
        
        # 准备样本文本和音频路径
        # 样本1：从1.txt读取文本内容，使用1.wav作为音频
        sample_text_1_path = os.path.join(data_dir, "1.txt")
        with open(sample_text_1_path, 'r', encoding='utf-8') as f:
            ref_text = f.read().strip()
        ref_audio_path = os.path.join(data_dir, "1.wav")
        
        # 确保参考音频文件存在
        ref_audio_path = Path(ref_audio_path).resolve()
        if not ref_audio_path.exists():
            raise FileNotFoundError(f"参考音频文件不存在: {ref_audio_path}")
        
        # 要生成的目标文本
        target_text = "你好，很高兴认识你。我的声音是用AI克隆的，听起来怎么样？"
        
        # 初始化预训练模型，而非微调模型（预训练模型更适合零样本克隆）
        print(f"正在加载模型: {model_name}")
        model = OrpheusModel(
            model_name=model_name,
            max_model_len=4096  # 增加上下文长度以容纳参考音频内容
        )
        
        print("参考音频：", ref_audio_path)
        print("参考文本：", ref_text)
        print("目标文本：", target_text)
        
        # 构建 DALLE 风格的零样本克隆提示格式
        # 对于预训练模型，使用参考文本和参考音频作为上下文，然后加入目标文本
        # 这种格式尝试让模型理解：这是参考文本的发音方式，现在用同样的声音说目标文本
        prompt = f"<speak>{ref_text}</speak> <audio>{ref_audio_path}</audio> <speak>{target_text}</speak>"
        
        print(f"使用以下提示进行语音生成：\n{prompt}")
        
        # 开始计时
        start_time = time.monotonic()
        
        # 生成语音（使用较高的重复惩罚和自定义温度以获得更好的韵律）
        print("开始生成语音...")
        syn_tokens = model.generate_speech(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=2000  # 足够长以包含完整响应
        )

        # 保存音频
        with wave.open(output_file, "wb") as wf:
            wf.setnchannels(1)  # 单声道
            wf.setsampwidth(2)  # 16位音频
            wf.setframerate(24000)  # 采样率24kHz

            total_frames = 0
            chunk_counter = 0
            for audio_chunk in syn_tokens:  # 流式输出
                chunk_counter += 1
                frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
                total_frames += frame_count
                wf.writeframes(audio_chunk)
            duration = total_frames / wf.getframerate()

            end_time = time.monotonic()
            print(f"生成 {duration:.2f} 秒的音频用时 {end_time - start_time:.2f} 秒")
            print(f"音频已保存至: {os.path.abspath(output_file)}")

    except Exception as e:
        print(f"生成过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 