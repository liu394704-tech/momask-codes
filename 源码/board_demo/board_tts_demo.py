import time
from hiwonder.TTS import TTS

'''
    程序功能：语音合成模块例程(program function: voice synthesis module routine)

    运行效果：程序启动后，语音识别模块将播放“你好”的声音一次(running effect: after the program starts, the speech recognition module will play the sound of 'hello' once)

    对应教程文档路径：  TonyPi智能视觉人形机器人\4.拓展课程学习\5.树莓派扩展板课程\第6课 树莓派语音合成(corresponding tutorial file path:TonyPi Intelligent Vision Humanoid Robot\4. Expanded Courses\5. Raspberry Pi Expansion Board\Lesson 6 Raspberry Pi Voice Synthesis)
'''

tts = TTS()

#[h0]设置单词发音方式，0为自动判断单词发音方式，1为字母发音方式，2为单词发音方式([h0]set the pronunciation mode for words: 0 for automatic pronunciation mode determination, 1 for letter-by-letter pronunciation mode, 2 for word-by-word pronunciation mode)
#[v10]设置音量，音量范围为0-10,10为最大音量。([v10]set the volume. The volume range is from 0 to 10, where 10 is the maximum volume)

#注意括号里的字符长度不能超过32,如果超过了请分多次来说(note that the character length inside the parentheses cannot exceed 32. If it exceeds, please split it into multiple parts)
tts.TTSModuleSpeak("[h0][v10]","你好")   

time.sleep(1) #要适当延时等说话完成(ensure to introduce appropriate delays to allow for speech completion)
