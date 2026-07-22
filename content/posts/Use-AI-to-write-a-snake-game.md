---
title: "利用 AI 写一个『贪吃蛇游戏』"
date: 2023-05-28T17:50:41+08:00
draft: false
tags: []
summary: "在完成一些简单的任务之后，我觉得 ChatGPT 可以帮我完成一些更为复杂，甚至于可能我并不是特别擅长的任务，而我想到了，也许，可以帮我写一个简单的游戏。"
---

贪多嚼不烂的道理我还是明白的，于是，我给自己提前就设定了一个目标：

> 游戏可以运行实现简单的规则减少 BUG 的存在可以开始和结束

带着这四个最朴素的愿景，让我们开始吧。

一贯的，一上来我就将问题抛给了 chatGPT

```plain
 请用 Python3.10 编写一个贪吃蛇的游戏，给我完整的代码。
```

这里我们需要注意一下，你的描述越清晰，得到的结果越有可能是你想要的。在这里，我告诉了 chatGPT，我希望用 Python 写一个游戏，而且版本号是 3.10。并且，我告诉了 chatGPT，请给我完整的代码。因为如果我不这么说的话，那很有可能 chatGPT 会先给我一堆的步骤而不是实际的代码。

在之后，我得到了 chatGPT 的答复：

![图片](https://qiniu.hivan.me/picGo/20230803223831.png?imgNote)

完整代码如下：

```python
 import pygame
 import random
 
 # 游戏窗口的大小
 WINDOW_WIDTH = 800
 WINDOW_HEIGHT = 600
 
 # 蛇身和食物的大小
 BLOCK_SIZE = 20
 
 # 定义颜色
 WHITE = (255, 255, 255)
 BLACK = (0, 0, 0)
 RED = (255, 0, 0)
 
 # 初始化 Pygame
 pygame.init()
 
 # 创建游戏窗口
 window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
 pygame.display.set_caption("贪吃蛇游戏")
 
 clock = pygame.time.Clock()
 
 def game_loop():
     game_over = False
 
     # 蛇的初始位置和速度
     snake_x = WINDOW_WIDTH // 2
     snake_y = WINDOW_HEIGHT // 2
     snake_x_change = 0
     snake_y_change = 0
 
     # 食物的初始位置
     food_x = round(random.randrange(0, WINDOW_WIDTH - BLOCK_SIZE) / BLOCK_SIZE) * BLOCK_SIZE
     food_y = round(random.randrange(0, WINDOW_HEIGHT - BLOCK_SIZE) / BLOCK_SIZE) * BLOCK_SIZE
 
     while not game_over:
         for event in pygame.event.get():
             if event.type == pygame.QUIT:
                 game_over = True
             elif event.type == pygame.KEYDOWN:
                 if event.key == pygame.K_LEFT:
                     snake_x_change = -BLOCK_SIZE
                     snake_y_change = 0
                 elif event.key == pygame.K_RIGHT:
                     snake_x_change = BLOCK_SIZE
                     snake_y_change = 0
                 elif event.key == pygame.K_UP:
                     snake_y_change = -BLOCK_SIZE
                     snake_x_change = 0
                 elif event.key == pygame.K_DOWN:
                     snake_y_change = BLOCK_SIZE
                     snake_x_change = 0
 
         # 更新蛇的位置
         snake_x += snake_x_change
         snake_y += snake_y_change
 
         # 绘制游戏窗口
         window.fill(BLACK)
         pygame.draw.rect(window, RED, [food_x, food_y, BLOCK_SIZE, BLOCK_SIZE])
         pygame.draw.rect(window, WHITE, [snake_x, snake_y, BLOCK_SIZE, BLOCK_SIZE])
         pygame.display.update()
 
         # 碰撞检测
         if snake_x == food_x and snake_y == food_y:
             food_x = round(random.randrange(0, WINDOW_WIDTH - BLOCK_SIZE) / BLOCK_SIZE) * BLOCK_SIZE
             food_y = round(random.randrange(0, WINDOW_HEIGHT - BLOCK_SIZE) / BLOCK_SIZE) * BLOCK_SIZE
 
         clock.tick(20)  # 控制游戏的帧率
 
     pygame.quit()
 
 # 启动游戏
 game_loop()
 
```

从代码里我们可以看出来，文件引用了库`pygame`以及`random`, 其中，`pygame`需要我们单独安装一下:

```plain
 conda install pygame
```

建立了一个`snake_v1.py`的文件，并把这段代码粘贴到文件里之后，我开始尝试运行它：

```plain
 python ~/xx/game/snake_v1.py
```

事情果然没有想的那么简单，根本就是毫无动静。我开始询问 chatGPT

![图片](https://qiniu.hivan.me/picGo/20230803223843.png?imgNote)

chatGPT 给到的答案似乎没有什么意义，这些事情在开始之前我就已经做好了。在短暂的思考之后，我忽然想到，是否因为我是 Mac 系统，从而导致了窗口不出现，于是我继续问问题：

![图片](https://qiniu.hivan.me/picGo/20230803223850.png?imgNote)

然后继续问：

![图片](https://qiniu.hivan.me/picGo/20230803223859.png?imgNote)

这一次，我抓到了重点。不明白为什么刚才我提到我是 Mac 系统的时候他不告诉我，先不管这些，在其中添加这段代码后，游戏终于可以运行了：

```python
 import pygame
 import random
 
 # 游戏窗口的大小
 WINDOW_WIDTH = 800
 WINDOW_HEIGHT = 600
 
 # 蛇身和食物的大小
 BLOCK_SIZE = 20
 
 # 定义颜色
 WHITE = (255, 255, 255)
 BLACK = (0, 0, 0)
 RED = (255, 0, 0)
 
 # 初始化 Pygame
 pygame.init()
 
 # 创建游戏窗口
 window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
 pygame.display.set_caption("贪吃蛇游戏")
 pygame.display.flip()
 
 clock = pygame.time.Clock()
 
 def game_loop():
     game_over = False
 
     # 蛇的初始位置和速度
     snake_x = WINDOW_WIDTH // 2
     snake_y = WINDOW_HEIGHT // 2
     snake_x_change = 0
     snake_y_change = 0
 
     # 食物的初始位置
     food_x = round(random.randrange(0, WINDOW_WIDTH - BLOCK_SIZE) / BLOCK_SIZE) * BLOCK_SIZE
     food_y = round(random.randrange(0, WINDOW_HEIGHT - BLOCK_SIZE) / BLOCK_SIZE) * BLOCK_SIZE
 
     while not game_over:
         for event in pygame.event.get():
             if event.type == pygame.QUIT:
                 game_over = True
             elif event.type == pygame.KEYDOWN:
                 if event.key == pygame.K_LEFT:
                     snake_x_change = -BLOCK_SIZE
                     snake_y_change = 0
                 elif event.key == pygame.K_RIGHT:
                     snake_x_change = BLOCK_SIZE
                     snake_y_change = 0
                 elif event.key == pygame.K_UP:
                     snake_y_change = -BLOCK_SIZE
                     snake_x_change = 0
                 elif event.key == pygame.K_DOWN:
                     snake_y_change = BLOCK_SIZE
                     snake_x_change = 0
 
         # 更新蛇的位置
         snake_x += snake_x_change
         snake_y += snake_y_change
 
         # 绘制游戏窗口
         window.fill(BLACK)
         pygame.draw.rect(window, RED, [food_x, food_y, BLOCK_SIZE, BLOCK_SIZE])
         pygame.draw.rect(window, WHITE, [snake_x, snake_y, BLOCK_SIZE, BLOCK_SIZE])
         pygame.display.update()
 
         # 碰撞检测
         if snake_x == food_x and snake_y == food_y:
             food_x = round(random.randrange(0, WINDOW_WIDTH - BLOCK_SIZE) / BLOCK_SIZE) * BLOCK_SIZE
             food_y = round(random.randrange(0, WINDOW_HEIGHT - BLOCK_SIZE) / BLOCK_SIZE) * BLOCK_SIZE
 
         clock.tick(20)  # 控制游戏的帧率
 
     pygame.quit()
 
 # 启动游戏
 game_loop()
 
```

![图片](https://qiniu.hivan.me/picGo/20230803223911.gif?imgNote)

不知道各位看出什么问题了没有，问题如下：

1. 动作太快了，完全是地狱级别的难度；

1. 蛇似乎不会增长，吃了物品没有变长；

1. 撞到墙壁后游戏虽然停止了，但是没任何提示，蛇就这么消失了；

于是，我只能让 chatGPT 再帮我优化代码：

![图片](https://qiniu.hivan.me/picGo/20230803223918.png?imgNote)

当然，又有了其他问题，

1. 蛇在撞到自己之后没有办法结束游戏

1. 游戏无法退出

1. 游戏无法重新开始

1. Game Over 之后显示的是乱码

当然，有编程经验的人大概能明白，乱码的话一定是因为中文显示不正确，而目前，我也没打算支持中文。干脆用英文好了。

于是，我又开始提需求，在连续三四轮这种询问之后，终于来到了最后一次：

![图片](https://qiniu.hivan.me/picGo/20230803223929.png?imgNote)

然后我得到了如下代码：

```python
 import pygame
 import random
 
 # 游戏窗口的大小
 WINDOW_WIDTH = 800
 WINDOW_HEIGHT = 600
 
 # 蛇身和食物的大小
 BLOCK_SIZE = 20
 
 # 定义颜色
 WHITE = (255, 255, 255)
 BLACK = (0, 0, 0)
 RED = (255, 0, 0)
 
 # 初始化 Pygame
 pygame.init()
 
 # 创建游戏窗口
 window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
 pygame.display.set_caption("贪吃蛇游戏")
 pygame.display.flip()
 
 clock = pygame.time.Clock()
 
 def game_loop():
     game_over = False
     game_restart = False
     game_paused = False
 
     while not game_restart:
         if game_over:
             show_game_over()
 
         # 蛇的初始位置和速度
         snake_x = WINDOW_WIDTH // 2
         snake_y = WINDOW_HEIGHT // 2
         snake_x_change = 0
         snake_y_change = 0
 
         # 蛇身的初始位置和长度
         snake_body = []
         snake_length = 1
 
         # 食物的初始位置
         food_x = round(random.randrange(0, WINDOW_WIDTH - BLOCK_SIZE) / BLOCK_SIZE) * BLOCK_SIZE
         food_y = round(random.randrange(0, WINDOW_HEIGHT - BLOCK_SIZE) / BLOCK_SIZE) * BLOCK_SIZE
 
         score = 0
 
         while not game_over:
             for event in pygame.event.get():
                 if event.type == pygame.QUIT:
                     pygame.quit()
                     quit()
                 elif event.type == pygame.KEYDOWN:
                     if event.key == pygame.K_LEFT and snake_x_change != BLOCK_SIZE:
                         snake_x_change = -BLOCK_SIZE
                         snake_y_change = 0
                     elif event.key == pygame.K_RIGHT and snake_x_change != -BLOCK_SIZE:
                         snake_x_change = BLOCK_SIZE
                         snake_y_change = 0
                     elif event.key == pygame.K_UP and snake_y_change != BLOCK_SIZE:
                         snake_y_change = -BLOCK_SIZE
                         snake_x_change = 0
                     elif event.key == pygame.K_DOWN and snake_y_change != -BLOCK_SIZE:
                         snake_y_change = BLOCK_SIZE
                         snake_x_change = 0
                     elif event.key == pygame.K_SPACE:
                         game_paused = not game_paused
                     elif event.key == pygame.K_q:
                         pygame.quit()
                         quit()
 
             if game_paused:
                 continue
 
             # 更新蛇的位置
             snake_x += snake_x_change
             snake_y += snake_y_change
 
             # 碰撞检测
             if snake_x == food_x and snake_y == food_y:
                 # 蛇吃到了食物，增加长度和得分
                 snake_length += 1
                 score += 1
                 food_x = round(random.randrange(0, WINDOW_WIDTH - BLOCK_SIZE) / BLOCK_SIZE) * BLOCK_SIZE
                 food_y = round(random.randrange(0, WINDOW_HEIGHT - BLOCK_SIZE) / BLOCK_SIZE) * BLOCK_SIZE
 
             # 更新蛇的身体
             snake_head = [snake_x, snake_y]
             snake_body.append(snake_head)
 
             if len(snake_body) > snake_length:
                 del snake_body[0]
 
             # 检测碰撞
             if snake_head in snake_body[:-1] or snake_x < 0 or snake_x >= WINDOW_WIDTH or snake_y < 0 or snake_y >= WINDOW_HEIGHT:
                 game_over = True
 
             # 绘制游戏窗口
             window.fill(BLACK)
             pygame.draw.rect(window, RED, [food_x, food_y, BLOCK_SIZE, BLOCK_SIZE])
             for body_part in snake_body:
                 pygame.draw.rect(window, WHITE, [body_part[0], body_part[1], BLOCK_SIZE, BLOCK_SIZE])
 
             # 显示实时分数
             font = pygame.font.Font(None, 28)
             text = font.render("Score: " + str(score), True, WHITE)
             window.blit(text, (10, 10))
 
             pygame.display.update()
             clock.tick(10)  # 控制游戏的帧率
 
     # 重新开始游戏
     game_loop()
 
 def show_game_over():
     font = pygame.font.Font(None, 36)
     text = font.render("Game Over. Press Space to restart", True, WHITE)
     text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
 
     while True:
         for event in pygame.event.get():
             if event.type == pygame.QUIT:
                 pygame.quit()
                 quit()
             elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                 game_loop()
 
         window.fill(BLACK)
         window.blit(text, text_rect)
         pygame.display.update()
         clock.tick(10)
 
 # 启动游戏
 game_loop()
```

大家可以看看效果：

![图片](https://qiniu.hivan.me/picGo/20230803223940.png?imgNote)

我们可以看到，这个游戏相对来说是比较完整了，速度正常了，有计分系统，蛇能正常增长，碰到自己游戏会结束，碰到墙壁后也会 Game Over，并且可以通过 SPACE 来重新开始。并且，我还加入了按 Q 键的时候退出的功能。

当然，还可以继续完善，比如随着时间的流逝，速度上可以渐渐加快等等。就看你怎么想，然后其他的交给 chatGPT。

通过这次的示例演示，其中重点不是教大家如何做一个贪吃蛇游戏，而是教大家如何利用 chatGPT 来解决你需要解决的问题。当然，我需要收回我开头说的话，chatGPT 并不能帮你解决你不熟悉的问题。就比如，如果我完全不懂这其中内容的话，可能我窗口都打不开，我完全都不知道我什么时候才能解决 Mac 系统中不一样的部分，而也正是因为有一些简单的经验，才让我考虑的那个层面，从而针对性提问解决了问题。

所以要记住，AI 并不能帮你解决你完全不懂的问题，起码，你要知道你想问什么，也要知道问题大概卡在哪里了，针对性继续提问。

最后，友情提示一下，不要用 API 来完成这一次次的对话，经验之谈，去买个 Plus，比 API 交互便宜多了。你看那一串串的代码重复的给你写出来，你完全不知道会耗费多少 Token。那些宝贵的 Token，还是用在聊天窗无法完成的任务上比较合适。