"""Generate an original synthetic diagram for a data-free inference smoke check."""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw
p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,default=Path('data/smoke/layout.png'))
a=p.parse_args()
a.output.parent.mkdir(parents=True,exist_ok=True)
im=Image.new('RGB',(600,800),'#e8dfcd')
d=ImageDraw.Draw(im)
d.rectangle((48,92,281,412),fill='#90b8cc',outline='white',width=12)
d.ellipse((109,155,210,259),fill='#e7b58e')
d.rectangle((114,258,204,383),fill='#326791')
d.rectangle((322,341,555,693),fill='#a3c891',outline='white',width=12)
d.polygon([(341,650),(421,440),(535,650)],fill='#426148')
d.ellipse((453,370,508,425),fill='#f6d367')
d.text((45,35),'COLLAGE SMOKE TEST',fill='#242424')
im.save(a.output)
print(a.output)
