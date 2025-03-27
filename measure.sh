#!/bin/bash

# Eingabedatei und Ausgabedatei definieren
INPUT_FILE="$1"
OUTPUT_FILE="$2"

# Dauer der Eingabedatei ermitteln
DURATION=$(sox --i -D "$INPUT_FILE")
if [ $? -ne 0 ]; then
  echo "Fehler: Konnte die Dauer der Datei nicht bestimmen."
  exit 1
fi

# Aufnahmezeit mit zusätzlicher Verzögerung berechnen
RECORD_DURATION=$(echo "$DURATION + 0.25" | bc)
RECORD_DURATION_INT=$(echo "$RECORD_DURATION" | awk '{print int($1+0.5)}')

# Abspielen und gleichzeitig aufnehmen
(aplay "$INPUT_FILE" & pid=$!
arecord -f cd -r 48000 -c 1 --duration=$(printf "%.0f" "$RECORD_DURATION_INT") "$OUTPUT_FILE"
wait $pid)
