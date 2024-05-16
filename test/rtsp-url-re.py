import re

# 定义 RTSP URL
rtsp_url = 'rtsp://admin:admin12345@192.168.6.64:554/Streaming/Channels/103?transportmode=unicast&profile=Profile_3'

# 使用正则表达式来提取用户名、密码和IP地址
pattern = re.compile(r'rtsp://([^:]+):([^@]+)@([^:/]+)')
match = pattern.match(rtsp_url)

if match:
    username = match.group(1)
    password = match.group(2)
    ip_address = match.group(3)

    print("用户名:", username)
    print("密码:", password)
    print("IP地址:", ip_address)
else:
    print("未找到匹配的用户名、密码和IP地址")