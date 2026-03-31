#!/bin/bash
set -e

echo "=== Schedule Engine Starting ==="
echo "  HTTP port: ${SCH_HTTP_PORT:-8100}"
echo "  gRPC port: ${SCH_GRPC_PORT:-50051}"

# Start gRPC server in background
python grpc_server.py &
GRPC_PID=$!

# Start HTTP server (foreground)
python http_server.py &
HTTP_PID=$!

echo "  gRPC PID: $GRPC_PID"
echo "  HTTP  PID: $HTTP_PID"

# Wait for either to exit
wait -n $GRPC_PID $HTTP_PID
EXIT_CODE=$?

# Terminate the other
kill $GRPC_PID $HTTP_PID 2>/dev/null || true
exit $EXIT_CODE
