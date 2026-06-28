"""Post-process the nbconvert HTML export into a clean, centered reading page.

Usage:  python tools/finalize_html.py <path-to-index.html>
Adds a small stylesheet (centered column, comfortable width) without touching the
notebook content. Safe to run repeatedly.
"""
import sys

CSS = """
<style>
  body.jp-Notebook { background: #ffffff; }
  body.jp-Notebook main {
      max-width: 1060px;
      margin: 0 auto;
      padding: 1.5rem 1.75rem 4rem;
  }
  /* a little more breathing room and readable line length for prose */
  .jp-RenderedHTMLCommon { font-size: 15px; line-height: 1.6; }
  .jp-RenderedHTMLCommon img { display: block; margin: 0.5rem auto; }
</style>
"""

def main(path):
    html = open(path, encoding="utf-8").read()
    if "</head>" in html and "body.jp-Notebook main" not in html:
        html = html.replace("</head>", CSS + "</head>", 1)
        open(path, "w", encoding="utf-8").write(html)
        print("finalized:", path)
    else:
        print("nothing to do (no </head> or already finalized):", path)

if __name__ == "__main__":
    main(sys.argv[1])
