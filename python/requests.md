## Requests





访问需要ssl证书的网站需要添加

## 下载大文件

```python
import requests
file_url = "http://www.xxxxx.com/"
r = requests.get(file_url, stream=True)
with open("xxx.png", "wb") as pdf:
    for chunk in r.iter_content(chunk_size=1024, verify=False): 
        if chunk:
            pdf.write(chunk)
 # 访问需要ssl证书的网站需要添加 verify=False	  
```

