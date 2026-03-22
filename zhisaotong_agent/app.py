import time
import datetime
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

import streamlit as st
from agent.react_agent import ReactAgent
from utils.history_manager import history_manager

# 页面配置
st.set_page_config(page_title="智扫通机器人智能客服", page_icon="🤖", layout="wide")

# --- 自定义 CSS 深度美化 ---
st.markdown("""
<style>
    /* 全局背景与字体 */
    .main {
        background-color: #f8f9fa;
    }
    
    /* 侧边栏样式 */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #eee;
    }
    
    /* 聊天气泡美化 */
    .stChatMessage {
        border-radius: 12px;
        padding: 5px 12px;
        margin-bottom: 12px;
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    /* 用户气泡 */
    [data-testid="stChatMessageUser"] {
        background-color: #e3f2fd;
    }
    
    /* 助手气泡 */
    [data-testid="stChatMessageAssistant"] {
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }

    /* 侧边栏按钮分类标题 */
    .sidebar-category {
        color: #888;
        font-size: 0.75rem;
        margin-top: 15px;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* 强制 Emoji 彩色显示 (Windows 优化) */
    .emoji-font {
        font-family: "Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji", sans-serif !important;
    }

    /* 隐藏 Header Anchor 🔗 */
    h1 a, h2 a, h3 a {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# 初始化 Session State
if "history_sessions" not in st.session_state:
    saved_sessions = history_manager.load_all_sessions()
    if saved_sessions:
        st.session_state["history_sessions"] = {sid: data["messages"] for sid, data in saved_sessions.items()}
        st.session_state["session_titles"] = {sid: data["title"] for sid, data in saved_sessions.items()}
        st.session_state["current_session_id"] = sorted(saved_sessions.keys(), reverse=True)[0]
    else:
        st.session_state["history_sessions"] = {}
        st.session_state["session_titles"] = {}

if "current_session_id" not in st.session_state:
    default_id = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    st.session_state["current_session_id"] = default_id
    st.session_state["history_sessions"][default_id] = [
        {"role": "assistant", "content": "你好，我是智扫通机器人智能客服，请问有什么可以帮助你？"}
    ]
    st.session_state["session_titles"][default_id] = "新对话"
    history_manager.save_session(default_id, "新对话", st.session_state["history_sessions"][default_id])

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

# --- 侧边栏渲染逻辑 ---
with st.sidebar:
    st.markdown("<h2 class='emoji-font'>🤖 智扫通</h2>", unsafe_allow_html=True)
    st.caption("您的专业扫地机器人管家")
    st.divider()

    if st.button("➕ 开启新对话", use_container_width=True, type="primary"):
        new_id = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        initial_msg = [{"role": "assistant", "content": "你好，我是智扫通机器人智能客服，请问有什么可以帮助你？"}]
        st.session_state["history_sessions"][new_id] = initial_msg
        st.session_state["session_titles"][new_id] = "新对话"
        st.session_state["current_session_id"] = new_id
        history_manager.save_session(new_id, "新对话", initial_msg)
        st.rerun()

    # 时间轴分组逻辑
    st.markdown('<p class="sidebar-category">对话记录</p>', unsafe_allow_html=True)
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    
    group_today, group_yesterday, group_older = [], [], []
    
    for s_id in sorted(st.session_state["history_sessions"].keys(), reverse=True):
        try:
            # 格式: session_20240315_221610
            date_str = s_id.split("_")[1]
            session_date = datetime.datetime.strptime(date_str, "%Y%m%d").date()
            if session_date == today:
                group_today.append(s_id)
            elif session_date == yesterday:
                group_yesterday.append(s_id)
            else:
                group_older.append(s_id)
        except:
            group_older.append(s_id)

    def render_group(ids, label):
        if ids:
            st.caption(label)
            for sid in ids:
                title = st.session_state["session_titles"].get(sid, "新对话")
                is_curr = (sid == st.session_state["current_session_id"])
                if st.button(f"💬 {title[:12]}...", key=sid, use_container_width=True, 
                             type="primary" if is_curr else "secondary"):
                    st.session_state["current_session_id"] = sid
                    st.rerun()

    render_group(group_today, "今天")
    render_group(group_yesterday, "昨天")
    render_group(group_older, "更早以前")

    st.divider()
    if st.button("🗑️ 清空当前对话", use_container_width=True):
        curr_id = st.session_state["current_session_id"]
        if len(st.session_state["history_sessions"]) > 1:
            history_manager.delete_session(curr_id)
            del st.session_state["history_sessions"][curr_id]
            del st.session_state["session_titles"][curr_id]
            st.session_state["current_session_id"] = list(st.session_state["history_sessions"].keys())[0]
            st.rerun()
        else:
            initial_msg = [{"role": "assistant", "content": "你好，我是智扫通机器人智能客服，请问有什么可以帮助你？"}]
            st.session_state["history_sessions"][curr_id] = initial_msg
            st.session_state["session_titles"][curr_id] = "新对话"
            history_manager.save_session(curr_id, "新对话", initial_msg)
            st.rerun()

# --- 主界面 ---
curr_id = st.session_state["current_session_id"]
messages = st.session_state["history_sessions"][curr_id]

# 首页欢迎布局
if len(messages) <= 1:
    st.markdown("<h2 style='text-align: center; margin-top: 50px;' class='emoji-font'>🤖 智扫通智能客服</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>专业的知识库驱动，为您解决所有扫地机难题</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.info("💡 **常见问题快捷查询：**")
    cols = st.columns(3)
    suggestions = ["扫地机器人不回充怎么办？", "如何清理滚刷上的头发？", "建图失败的解决方法"]
    for i, ques in enumerate(suggestions):
        if cols[i].button(ques, use_container_width=True):
            st.session_state["auto_prompt"] = ques
            st.rerun()
    st.markdown("---")
else:
    curr_title = st.session_state["session_titles"].get(curr_id, "新对话")
    st.markdown(f"<h1 class='emoji-font'>💬 {curr_title}</h1>", unsafe_allow_html=True)
    st.divider()

# 辅助输入逻辑
auto_prompt = st.session_state.pop("auto_prompt", None)
prompt_to_use = st.chat_input("向我提问...") if not auto_prompt else auto_prompt

# 渲染消息
for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 处理交互
if prompt_to_use:
    if st.session_state["session_titles"][curr_id] == "新对话":
        st.session_state["session_titles"][curr_id] = prompt_to_use[:12]

    with st.chat_message("user"):
        st.markdown(prompt_to_use)
    messages.append({"role": "user", "content": prompt_to_use})
    history_manager.save_session(curr_id, st.session_state["session_titles"][curr_id], messages)

    with st.chat_message("assistant"):
        res_box = st.empty()
        full_res = ""
        with st.spinner("思考中..."):
            for chunk in st.session_state["agent"].execute_stream(prompt_to_use):
                full_res += chunk
                res_box.markdown(full_res + " ●")
                time.sleep(0.01)
            res_box.markdown(full_res)
    
    messages.append({"role": "assistant", "content": full_res})
    history_manager.save_session(curr_id, st.session_state["session_titles"][curr_id], messages)
    st.rerun()
