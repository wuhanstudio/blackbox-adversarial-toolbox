#!/usr/bin/env python3
from bat import __version__

import shutil
from pathlib import Path

import pdoc

here = Path(__file__).parent

modules = [
    "bat",
]

print('Building docs for v{}'.format(__version__))

# Render pdoc's documentation into docs/api...
pdoc.render.configure(
    template_directory = here / "pdoc-dark-mode", 
    footer_text="Black-box Adversarial Toolbox v" + str(__version__),
    search=True,
    logo="https://bat.wuhanstudio.uk/images/bat_dark.png",
    logo_link="https://github.com/wuhanstudio/blackbox-adversarial-toolbox")

pdoc.pdoc(*modules, output_directory = here / "docs")


from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print("Encountered a start tag:", tag)

    def handle_endtag(self, tag):
        print("Encountered an end tag :", tag)

    def handle_data(self, data):
        print("Encountered some data  :", data)

print('Generating index.html')

# homepage ='docs/index.html' 
# with open(homepage, 'w') as filetowrite:
#     filetowrite.write('<!DOCTYPE html><html><head><script type="text/javascript">')
#     filetowrite.write('window.location.href = window.location.href + "v')
#     filetowrite.write(str(__version__))
#     filetowrite.write('" + "/bat.html";')
#     filetowrite.write('</script></head><body></body></html>')
