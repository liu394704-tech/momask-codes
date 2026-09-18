import time
import ros_robot_controller_sdk as rrc

'''
    程序功能：IMU例程(MPU6050)(program function: IMU routine(MPU6050))

    运行效果：程序运行后，在屏幕不断输出获取到的IMU的数据(running effect: after the program runs, the IMU data is continuously outputted on the screen)

    对应教程文档路径：  TonyPi智能视觉人形机器人\4.拓展课程学习\5.树莓派扩展板课程\第7课 加速度计的使用(corresponding tutorial file path:TonyPi Intelligent Vision Humanoid Robot\4. Expanded Courses\5. Raspberry Pi Expansion Board\Lesson 7 The Use of Accelerometer)
'''
board = rrc.Board()
board.enable_reception()

while True:
    try:
        res = board.get_imu()          # 获取IMU数据(get IMU data)
        if res is not None:
            print(res)           # 输出获取到的IMU数据(ouput the obtained IMU data)
        
        time.sleep(0.01)
    except KeyboardInterrupt:
        break
