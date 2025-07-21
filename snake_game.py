#!/usr/bin/env python3
"""
Snake engine — classic wall collisions + head‑less training helper.

Returns richer rewards:
    score      = # food eaten
    ticks      = time survived
    composite  = ticks + 10 * score       (smoother signal)
"""

import random, time, importlib, sys

BOARD_W, BOARD_H = 10, 10
TICK_SEC          = 0.05
MAX_TICKS         = 1_000

DIR_VEC  = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}

# --------------------------- Snake engine class ------------------------- #
class SnakeGame:
    def __init__(self, width, height):
        self.w, self.h = width, height
        cx, cy = width // 2, height // 2
        self.snake     = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self.direction = "RIGHT"
        self.food      = self._spawn_food()
        self.score     = 0
        self.ticks     = 0
        self.alive     = True

    def _spawn_food(self):
        free = [(x, y) for x in range(self.w) for y in range(self.h)
                if (x, y) not in self.snake]
        return random.choice(free)

    def _next_head(self, direction):
        vx, vy = DIR_VEC[direction]
        hx, hy = self.snake[0]
        return hx + vx, hy + vy          # classic — no wrap

    def step(self, direction):
        if direction == OPPOSITE[self.direction]:
            direction = self.direction
        self.direction = direction

        new_head = self._next_head(direction)

        x, y = new_head
        if not (0 <= x < self.w and 0 <= y < self.h) or new_head in self.snake:
            self.alive = False
            return

        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.score += 1
            self.food = self._spawn_food()
        else:
            self.snake.pop()

    # ---------- rendering ----------
    def draw(self):
        board = [[" "] * self.w for _ in range(self.h)]
        fx, fy = self.food
        board[fy][fx] = "*"
        for i, (x, y) in enumerate(self.snake):
            board[y][x] = "@" if i == 0 else "o"

        top = "+" + "-" * self.w + "+"
        print("\x1b[H", end="")
        print(top)
        for row in board:
            print("|" + "".join(row) + "|")
        print(top)
        print(f"Score: {self.score}  Tick: {self.ticks}")

# ----------------------- Convenience function -------------------------- #
def run_episode(bot, render=False, width=BOARD_W, height=BOARD_H,
                max_ticks=MAX_TICKS):
    """
    Plays one game and returns a dict:
        {"score": int, "ticks": int, "composite": int}
    composite = ticks + 10 * score
    """
    game = SnakeGame(width, height)

    if render:
        print("\x1b[2J", end="")   # clear once

    while game.alive and game.ticks < max_ticks:
        state = {
            "board_width":  game.w,
            "board_height": game.h,
            "snake":        tuple(game.snake),
            "food":         game.food,
            "direction":    game.direction,
            "tick":         game.ticks,
            "score":        game.score,
        }

        try:
            move = bot.next_move(state)
            if move not in DIR_VEC:
                raise ValueError
        except Exception as e:
            print(f"[Bot error fallback] {e}")
            move = game.direction

        game.step(move)
        game.ticks += 1

        if render:
            game.draw()
            time.sleep(TICK_SEC)

    result = {
        "score":     game.score,
        "ticks":     game.ticks,
        "composite": game.ticks + 10 * game.score,
    }

    if render:
        print("\n=== GAME OVER ===")
        print(f"Score    : {result['score']}")
        print(f"Ticks    : {result['ticks']}")
        print(f"Composite: {result['composite']}")

    return result

# ----------------------- Stand‑alone play helper ----------------------- #
def _load_bot():
    if "bot" in sys.modules:
        importlib.reload(sys.modules["bot"])
    else:
        import bot
    return sys.modules["bot"].SnakeBot()

def main():
    bot = _load_bot()
    run_episode(bot, render=True)

if __name__ == "__main__":
    main()
