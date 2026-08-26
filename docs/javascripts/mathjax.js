window.MathJax = {
  loader: {
    load: ["[tex]/ams"]
  },
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    packages: { "[+]": ["ams"] },
    tags: "ams",
    tagSide: "right",
    tagIndent: "0.8em"
  },
  chtml: {
    scale: 1,
    mtextInheritFont: true,
    displayAlign: "center",
    displayIndent: "0"
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex|equation-context"
  }
};

document$.subscribe(() => {
  if (!window.MathJax || !MathJax.typesetPromise) {
    return;
  }
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
