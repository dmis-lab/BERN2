#!/bin/bash

pid=`ps auxww | grep ner_server.py | grep -v grep | awk '{print $2}' | sort -r`
if [ "$pid" != "" ]; then
  kill -9 "$pid"
  echo "Stopped ner_server.py"
else
  echo "No ner_server.py found to stop."
fi

pid=`ps auxww | grep GNormPlusServer.main.jar | grep -v grep | awk '{print $2}' | sort -r`
if [ "$pid" != "" ]; then
  kill -9 "$pid"
  echo "Stopped GNormPlusServer.main.jar"
else
  echo "No GNormPlusServer.main.jar found to stop."
fi

pid=`ps auxww | grep tmVar2Server.main.jar | grep -v grep | awk '{print $2}' | sort -r`
if [ "$pid" != "" ]; then
  kill -9 "$pid"
  echo "Stopped tmVar2Server.main.jar"
else
  echo "No tmVar2Server.main.jar found to stop."
fi

pid=`ps auxww | grep disease_normalizer_21.jar | grep -v grep | awk '{print $2}' | sort -r`
if [ "$pid" != "" ]; then
  kill -9 "$pid"
  echo "Stopped disease_normalizer_21.jar"
else
  echo "No disease_normalizer_21.jar found to stop."
fi

pid=`ps auxww | grep gnormplus-normalization_21.jar | grep -v grep | awk '{print $2}' | sort -r`
if [ "$pid" != "" ]; then
  kill -9 "$pid"
  echo "Stopped gnormplus-normalization_21.jar"
else
  echo "No gnormplus-normalization_21.jar found to stop."
fi

pid=`ps auxww | grep server.py | grep -v grep | awk '{print $2}' | sort -r`
if [ "$pid" != "" ]; then
  kill -9 "$pid"
  echo "Stopped server.py"
else
  echo "No server.py found to stop."
fi