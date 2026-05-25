import os
import json
from datetime import datetime
from typing import Any, Dict, List


class XAILogger:
    def __init__(self, option_name: str, out_dir: str):
        self.option_name = option_name
        self.out_dir = out_dir
        self.steps: List[Dict[str, Any]] = []
        os.makedirs(self.out_dir, exist_ok=True)

    def add_step(self, title: str, details: Dict[str, Any]):
        self.steps.append({
            "title": title,
            "details": details,
            "ts": datetime.utcnow().isoformat() + "Z",
        })

    def _json_path(self) -> str:
        return os.path.join(self.out_dir, f"xai_report_{self.option_name}.json")

    def _md_path(self) -> str:
        return os.path.join(self.out_dir, f"xai_report_{self.option_name}.md")

    def write(self):
        # JSON
        payload = {
            "option": self.option_name,
            "created_utc": datetime.utcnow().isoformat() + "Z",
            "steps": self.steps,
        }
        with open(self._json_path(), 'w') as f:
            json.dump(payload, f, indent=2)
        # Markdown
        lines = [f"# XAI Report - {self.option_name}", ""]
        for i, s in enumerate(self.steps, start=1):
            lines.append(f"## {i}. {s['title']}")
            for k, v in s.get('details', {}).items():
                lines.append(f"- {k}: {v}")
            lines.append("")
        with open(self._md_path(), 'w') as f:
            f.write("\n".join(lines))

    def paths(self) -> Dict[str, str]:
        return {"json": self._json_path(), "md": self._md_path()}
