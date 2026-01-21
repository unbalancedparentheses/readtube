{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    # Python
    python312

    # WeasyPrint system dependencies
    pango
    cairo
    gdk-pixbuf
    gobject-introspection
    glib
    fontconfig
    freetype
    harfbuzz

    # Build tools
    pkg-config

    # Optional: Local LLM tooling
    ollama
    llama-cpp
    python312Packages.llama-cpp-python
    python312Packages.anthropic
    python312Packages.openai

    # Fonts
    liberation_ttf
    dejavu_fonts
  ];

  shellHook = ''
    echo "Readtube development environment"
    echo "Run 'make install' to install Python dependencies"

    # Set up library paths for WeasyPrint
    export GI_TYPELIB_PATH="${pkgs.pango}/lib/girepository-1.0:${pkgs.gdk-pixbuf}/lib/girepository-1.0"
  '';
}
