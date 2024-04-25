import nmap

# 创建nmap扫描器对象
nm = nmap.PortScanner()

# 扫描局域网中的设备和端口
nm.scan(hosts='192.168.2.0/24', arguments='-p 554')

# 遍历扫描结果
for host in nm.all_hosts():
    if nm[host].has_tcp(554) and nm[host]['tcp'][554]['state'] == 'open':
        print(f"发现开放的RTSP端口在主机 {host}")