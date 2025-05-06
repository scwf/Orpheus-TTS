from orpheus_tts import OrpheusModel
import wave
import time

if __name__ == "__main__":  # 添加主程序入口保护
    try:
        # 初始化模型，并设置较小的 max_model_len
        model = OrpheusModel(
            # 中文模式使用：canopylabs/3b-zh-ft-research_release、canopylabs/3b-zh-pretrain-research_release
            model_name="canopylabs/3b-zh-ft-research_release",
            max_model_len=2048
        )
        
        # 准备提示文本
        prompt = "你好，小鱼儿和小贝壳，我是机器人小丽。"
        
        # 开始计时
        start_time = time.monotonic()
        
        # 生成语音
        syn_tokens = model.generate_speech(
            prompt=prompt,
            voice="白芷"
            # repetition_penalty=1.1,  # 必需：确保稳定生成
            # stop_token_ids=[128258],  # 结束标记
            # temperature=0.4,  # 控制生成随机性
            # top_p=0.9,  # 控制采样范围
            # max_tokens=2000  # 防止生成过长
        )

        # 保存音频
        with wave.open("output.wav", "wb") as wf:
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
            print(f"生成 {duration:.2f} 秒的音频用时 {end_time - start_time} 秒")

    except Exception as e:
        print(f"生成过程中出现错误: {str(e)}")
