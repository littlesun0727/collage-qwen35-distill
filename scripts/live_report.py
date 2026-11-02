"""Local results-page writer; does not connect to or supervise the GPU server."""
import argparse
import html
import json
from pathlib import Path
import time

def build(root):
    summary = json.loads((root / 'batch-summary.json').read_text(encoding='utf-8'))
    rows = summary['cases']
    complete = sum(row['status'] == 'complete' for row in rows)
    cards = []
    for row in rows:
        base = 'cases/' + row['id']
        pictures = ''.join('<a href="%s/%s"><img loading="lazy" src="%s/%s"></a>' % (base, name, base, name)
                           for name in ('original.jpg', 'bbox.jpg') if (root / base / name).exists())
        links = ' · '.join('<a href="%s/%s">%s</a>' % (base, name, name)
                           for name in ('structure.json', 'analysis.json', 'case-summary.json') if (root / base / name).exists())
        cards.append('<article><h3>%s · %s</h3>%s<p>%s</p><pre>%s</pre></article>' %
                     (html.escape(row['relative_path']), row['status'], pictures, links, html.escape(row.get('error', ''))))
    provenance = json.loads((root / 'provenance.json').read_text(encoding='utf-8'))
    model = provenance['policy']['analyze']['model']
    title = '%s · covers · %d/%d' % (model, len(rows), summary['total'])
    page = '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="20"><title>' + title + '</title>'
    page += '<style>body{font-family:system-ui;margin:24px;background:#f4f5f7}article{background:white;padding:16px;margin:20px 0}img{max-width:46%;max-height:600px;margin:1%}pre{white-space:pre-wrap}h1{position:sticky;top:0;background:#f4f5f7;padding:12px}</style>'
    page += '<h1>' + title + '</h1><p>完成 %d，失败 %d。左侧原图，右侧模型框图。每 20 秒刷新；校验通过不等于识别正确。</p>' % (complete, len(rows)-complete)
    page += ''.join(cards)
    temporary = root / 'live.tmp.html'
    temporary.write_text(page, encoding='utf-8')
    temporary.replace(root / 'live.html')
    return len(rows) >= summary['total']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        done = build(args.root)
        if done or not args.watch:
            break
        time.sleep(10)
