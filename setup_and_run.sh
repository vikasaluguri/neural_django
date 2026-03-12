#!/bin/bash
# ─────────────────────────────────────────────
#  NeuralForge — One-shot setup & run script
# ─────────────────────────────────────────────
set -e

echo ""
echo "══════════════════════════════════════"
echo "  NeuralForge — Neural Network Django "
echo "══════════════════════════════════════"
echo ""

# 1. Install dependencies
echo "[ 1/4 ] Installing dependencies..."
pip install -r requirements.txt --quiet

# 2. Migrations
echo "[ 2/4 ] Running database migrations..."
python manage.py makemigrations --verbosity 0
python manage.py migrate --verbosity 0

# 3. Create superuser (optional, non-blocking)
echo "[ 3/4 ] Creating admin superuser (optional)..."
echo "        (Press CTRL+C to skip, or type credentials below)"
python manage.py createsuperuser --username admin 2>/dev/null || echo "        Skipped — admin already exists or input cancelled."

# 4. Run server
echo ""
echo "[ 4/4 ] Starting development server..."
echo ""
echo "  → Open: http://127.0.0.1:8000/"
echo "  → Admin: http://127.0.0.1:8000/admin/"
echo ""
python manage.py runserver
