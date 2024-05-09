在Git中，你可以通过使用`git submodule`命令来引入一个子仓库。子模块允许你将一个Git仓库作为另一个Git仓库的子目录。这在管理外部依赖或共享代码时非常有用。下面是具体如何操作的步骤：

1. **打开终端**：首先，打开你的终端（或命令提示符），并切换到你的主仓库的根目录。

2. **添加子模块**：使用`git submodule add`命令来添加子模块。你需要提供子模块的URL以及你希望子模块被克隆到的目录。对于你的例子，命令如下：

   ```bash
   git submodule add git@github.com:hammershock/NumPyTorch.git NumPyTorch
   ```

   这条命令会在`NumPyTorch`目录中克隆子仓库，并在你的主仓库中添加必要的配置信息。

3. **初始化子模块**：如果这是你第一次在仓库中添加子模块，你需要初始化子模块配置。可以通过以下命令来完成：

   ```bash
   git submodule init
   ```

4. **更新子模块**：为确保子模块代码是最新的，执行以下命令：

   ```bash
   git submodule update
   ```

5. **提交更改**：添加子模块后，你的主仓库会有新的更改。需要提交这些更改：

   ```bash
   git add .
   git commit -m "Added NumPyTorch submodule"
   git push
   ```

6. **克隆包含子模块的仓库**：当其他人或者你自己在另一台机器上克隆含有子模块的仓库时，需要使用特定的命令来同时克隆主仓库和子模块：

   ```bash
   git clone --recurse-submodules <repository-url>
   ```

   如果已经克隆了主仓库而没有克隆子模块，可以使用以下命令来更新子模块：

   ```bash
   git submodule update --init --recursive
   ```

以上步骤会帮助你在Git仓库中正确地引入和管理子仓库。
