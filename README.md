# 黑马大模型 RAG 与 Agent 实战学习笔记

本仓库记录了 B 站黑马程序员课程 **《黑马程序员大模型 RAG 与 Agent 智能体项目实战教程》** 的学习过程与代码实现。

[![Course Link](https://img.shields.io/badge/Bilibili-%E8%AF%BE%E7%A8%8B%E9%93%BE%E6%8E%A5-blue)](https://www.bilibili.com/video/BV1yjz5BLEoY)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-green)](https://www.python.org/)

---

## 📂 项目结构 Directory Structure

### 1. 基础学习与 LangChain 练习 (`AI_LLM_RAG_Agent_Dev/`)
包含从基础调用到复杂 Agent 编排的 30+ 个实验脚本。为了简洁，此处按功能分类列出核心内容：

- **基础调用与算法**:
  - `01`-`04`: API Key 测试、SDK 基础用法、流式输出、会话历史。
  - `05`-`09`: 金融分类、JSON 提取、Few-shot 学习等实战案例。
  - `10`: 余弦相似度算法手写实现。
- **LangChain 核心组件**:
  - `11`-`14`: 基础用法、消息对象封装。
  - `15`: 向量化 (DashScope Embeddings)。
  - `16`-`19`: PromptTemplate、ChatPromptTemplate 及动态注入。
  - `20`-`22`: LCEL (LangChain Expression Language) 链式调用与机制分析。
- **进阶功能**:
  - `23`-`25`: 输出解析器 (Str/Json OutputParser) 与 RunnableLambda。
  - `26`-`27`: 内存/持久化会话记忆。
  - `28`-`31`: 数据加载器 (CSV/JSON/Text/PDF)。
- **RAG & Agent 实战**:
  - `32`-`34`: 向量存储 (Chroma) 与完整 RAG 工作流。
  - `35`-`38`: Agent 智能体初体验、流式对话、ReAct 框架及中间件拦截。

### 2. 服装电商智能客服 (`rag-clothing-customer-service/`)
基于 RAG 技术的服装电商智能客服系统。
- **核心功能**: 尺码推荐、洗涤养护、颜色选择问答。
- **技术栈**: Chroma + LangChain + Streamlit。
- **启动方式**: 运行 `run_qa.sh` (或 `.bat`) 开启问答界面。

![RAG 服装客服问答助手](data/RAG%20服装客服问答助手.png)
![知识库更新服务](data/知识库更新服务.png)

### 3. 智扫通机器人智能客服 (`zhisaotong_agent/`)  `New!`
一个面向消费者的专业扫地机器人管家系统。
- **定位**: 提供全生命周期的导购与售后支持。
- **亮点**:
  - **ReAct 架构**: 结合“思考-行动-观察”闭环，精准调用工具。
  - **深度 RAG**: 针对扫地机领域知识库进行增强。
  - **美化界面**: 使用 Streamlit 打造的现代化聊天交互界面。
- **主要入口**: `zhisaotong_agent/app.py`。

![智扫通智能客服1](data/智扫通智能客服1.png)
![智扫通智能客服2](data/智扫通智能客服2.png)

---

## 🚀 环境与运行 Environment & How to Run

### 1. 配置环境变量
复制示例配置文件并填写你的 **阿里云 DashScope API Key**：
```bash
cp .env.example .env
```
> [!IMPORTANT]
> 即使变量名为 `API_KEY`，也必须使用 DashScope 密钥。本项目深度依赖通义千问系列模型。

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 运行子项目
- **运行练习脚本**: `python AI_LLM_RAG_Agent_Dev/11_LangChain_Tongyi_Basic_Usage.py`
- **启动服装客服**: `cd rag-clothing-customer-service && streamlit app_qa.py`
- **启动智扫通 Agent**: `cd zhisaotong_agent && streamlit run app.py`

---

## 📜 声明 Disclaimer
本项目仅用于个人学习，与黑马程序员官方无直接关联。欢迎开发者参考、交流与扩展。

- 学习用途：本仓库仅用于个人学习与笔记整理，无任何商业用途。
- 非官方代码：本项目与黑马程序员、课程官方无直接关联，仅参考其公开课程内容进行实践。
- 欢迎扩展：你可以在此基础上继续扩展自己的 RAG / Agent 实战项目与实验。

