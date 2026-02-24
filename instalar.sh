#!/bin/bash
echo "╔══════════════════════════════════════╗"
echo "║   InfraInvestor — Instalando...      ║"
echo "╚══════════════════════════════════════╝"

pip install -r requirements.txt

echo ""
echo "✅ Instalação concluída!"
echo ""
echo "Para rodar o bot:"
echo "  python bot.py"
echo ""
echo "Dashboard web disponível em:"
echo "  http://localhost:8080"
echo ""
echo "Comandos Telegram disponíveis:"
echo "  /status   /radar   /historico"
echo "  /backtest /ml      /pausar"
echo "  /retomar  /meta    /ajuda"
