# 使用说明

## 🚀 快速开始

### 1. 本地查看文档

```bash
# 方式一：使用启动脚本（推荐）
./start.sh

# 方式二：直接使用Python
python3 -m http.server 8000
```

然后访问 http://localhost:8000

### 2. 更新文档

直接编辑对应的 `.md` 文件即可，网站会自动更新：

- **执行摘要**: `README.md`
- **技术文档**: `02-technology/` 目录下的文件
- **商业计划**: `03-business/` 和 `05-roadmap/` 目录

### 3. 部署到 GitHub Pages

1. 将代码推送到 GitHub 仓库
2. 在仓库设置中启用 GitHub Pages
3. 选择从根目录部署
4. 访问 `https://[用户名].github.io/[仓库名]/`

### 4. 添加新文档

1. 创建新的 `.md` 文件
2. 在 `_sidebar.md` 中添加链接
3. 刷新页面即可看到

## 📝 Markdown 技巧

- **标题**: 使用 `#`, `##`, `###`
- **强调**: 使用 `**粗体**` 和 `*斜体*`
- **代码**: 使用 ``` 包围代码块
- **链接**: `[文字](链接)`
- **图片**: `![描述](图片地址)`

## 🔧 自定义

如需修改网站配置，编辑 `index.html` 中的 `window.$docsify` 对象。 