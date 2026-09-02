#!/bin/bash
# vexp-restore: context lifecycle restore on SessionStart (compact/resume). Fails open.
VEXP_BIN="c:/Users/de larrard alexandre/.vscode/extensions/vexp.vexp-vscode-3.1.0-win32-x64/binaries/vexp-core-win32-x64/vexp-core.exe"
[ -x "$VEXP_BIN" ] || exit 0
"$VEXP_BIN" session-context 2>/dev/null
exit 0
