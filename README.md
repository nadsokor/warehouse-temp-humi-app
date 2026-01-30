# 温湿度数据处理 Web App

本项目将本地脚本封装为一个简易 Web 应用，支持上传 Excel、设置参数、下载处理结果。

## 本地运行

1. 安装依赖：
   - `pip install -r requirements.txt`
2. 启动应用：
   - `streamlit run app.py`
3. 浏览器访问：
   - 终端会提示本地地址，例如 `http://localhost:8501`

## 输入要求

Excel 需包含以下列名（与脚本一致）：

- `采集时间`
- `仪表名称`
- `仪表编号`
- `管理主机编号`
- `温度℃`
- `湿度%RH`
