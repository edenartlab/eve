#!/bin/bash
# Script to run observability tests

set -e

echo "=================================================="
echo "Running Prompt Session Observability Tests"
echo "=================================================="

# Check environment
if [ "$ENV" != "staging" ]; then
    echo "⚠️  Warning: ENV is not set to 'staging'"
    echo "   Current ENV: ${ENV:-not set}"
    echo ""
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for required environment variables
echo ""
echo "Environment Check:"
echo "------------------"

if [ -n "$SENTRY_DSN" ]; then
    echo "✓ SENTRY_DSN is configured"
else
    echo "✗ SENTRY_DSN is not configured"
fi

if [ -n "$LANGFUSE_PUBLIC_KEY" ] && [ -n "$LANGFUSE_SECRET_KEY" ]; then
    echo "✓ LANGFUSE keys are configured"
else
    echo "✗ LANGFUSE keys are not configured"
fi

if [ -n "$MONGO_URI" ]; then
    echo "✓ MONGO_URI is configured"
else
    echo "✗ MONGO_URI is not configured"
fi

echo ""

# Menu
echo "Select test to run:"
echo "1) Unit tests (pytest)"
echo "2) Integration test (staging DB)"
echo "3) Both"
echo ""
read -p "Choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "Running unit tests..."
        echo "===================="
        python -m pytest tests/session/test_observability.py -v
        ;;
    2)
        echo ""
        echo "Running integration test..."
        echo "=========================="
        python tests/session/test_observability_integration.py
        ;;
    3)
        echo ""
        echo "Running unit tests..."
        echo "===================="
        python -m pytest tests/session/test_observability.py -v

        echo ""
        echo "Running integration test..."
        echo "=========================="
        python tests/session/test_observability_integration.py
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=================================================="
echo "Tests completed!"
echo "=================================================="