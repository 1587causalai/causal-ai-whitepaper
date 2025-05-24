#!/bin/bash

echo "🚀 启动 Causal AI Whitepaper 文档网站..."
echo ""
echo "请选择启动方式："
echo "1) Python HTTP Server (推荐，无需安装)"
echo "2) Docsify CLI (需要先安装: npm i docsify-cli -g)"
echo ""
read -p "请输入选择 (1 或 2): " choice

case $choice in
    1)
        echo "使用 Python 启动..."
        echo "网站地址: http://localhost:8000"
        python3 -m http.server 8000
        ;;
    2)
        echo "使用 Docsify 启动..."
        echo "网站地址: http://localhost:3000"
        docsify serve .
        ;;
    *)
        echo "无效选择，默认使用 Python"
        python3 -m http.server 8000
        ;;
esac 