import json
import os
from datetime import datetime
from typing import Dict, List, Any

class HistoryManager:
    def __init__(self, storage_dir: str = "data/history"):
        self.storage_dir = storage_dir
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

    def _get_session_path(self, session_id: str) -> str:
        return os.path.join(self.storage_dir, f"{session_id}.json")

    def save_session(self, session_id: str, title: str, messages: List[Dict[str, Any]]):
        data = {
            "session_id": session_id,
            "title": title,
            "messages": messages,
            "updated_at": datetime.now().isoformat()
        }
        with open(self._get_session_path(session_id), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_session(self, session_id: str) -> Dict[str, Any]:
        path = self._get_session_path(session_id)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def load_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        sessions = {}
        for filename in os.listdir(self.storage_dir):
            if filename.endswith(".json"):
                session_id = filename[:-5]
                data = self.load_session(session_id)
                if data:
                    sessions[session_id] = data
        return sessions

    def delete_session(self, session_id: str):
        path = self._get_session_path(session_id)
        if os.path.exists(path):
            os.remove(path)

history_manager = HistoryManager()
