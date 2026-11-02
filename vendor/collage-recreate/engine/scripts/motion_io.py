"""Atomic motion evidence writes tolerate Windows readers briefly holding a file."""
import json
import os
from pathlib import Path
import time
import uuid


def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex[:12]+'.tmp')
    try:
        temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
        for attempt in range(9):
            try:
                os.replace(temporary,path);return
            except PermissionError:
                if attempt==8:raise
                time.sleep(min(.005*2**attempt,.15))
    finally:
        temporary.unlink(missing_ok=True)
