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

    # Start Ollama server in the background if not already running
    if ! curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
      echo "Starting Ollama server..."
      ollama serve > /dev/null 2>&1 &
      export OLLAMA_PID=$!
      # Wait briefly for it to be ready
      for i in $(seq 1 10); do
        if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
          echo "Ollama server ready (PID $OLLAMA_PID)"
          break
        fi
        sleep 0.5
      done
    else
      echo "Ollama server already running"
    fi
  '';
}
