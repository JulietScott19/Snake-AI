# bot.py  –  full upgraded SnakeBot
import tensorflow as tf
import numpy as np
import random, sys, pathlib, argparse, matplotlib.pyplot as plt
from collections import deque

# ----------------- hyper‑params -----------------
GAMMA           = 0.99          # cares 1% less about future rewards than current rewards
PER_TICK_PENALTY= -0.01         # discourages idling
FOOD_REWARD     =  1.0
DIST_BONUS      =  0.2          # +0.2 if closer, ‑0.2 if further

# ----------------- helper -----------------
def manhattan(ax, ay, bx, by): return abs(ax-bx)+abs(ay-by) #using manhattan distance

class SnakeBot:
    DIRECTIONS        = ["UP","DOWN","LEFT","RIGHT"]
    IDX_TO_DIRECTION  = dict(enumerate(DIRECTIONS))
    DIRECTION_TO_IDX  = {d:i for i,d in IDX_TO_DIRECTION.items()}

    def __init__(self):
        self.model = self._build_model()
        self.opt   = tf.keras.optimizers.Adam(1e-3)

        self.epsilon        = 0.20 #starts with 20% random moves to stop early convergence
        self.epsilon_min    = 0.01 #lowest amount of randomness
        self.epsilon_decay  = 0.995 # decay each game

        self.reset_ep()
        self.ep            = 0 
        self.avg_return    = 0 
        self.ret_history   = []

    # -------- network ----------
    def _build_model(self):
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=(11,)), 
            tf.keras.layers.Dense(128),
            tf.keras.layers.LeakyReLU(), #like normal ReLU but keeps a small gradient for negatice inputs to reduce the chance of death
            tf.keras.layers.Dense(64),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(4)          # logits
        ])

    # -------- geometry helpers ----------
    def _left_of(self,d):  return {"UP":"LEFT","LEFT":"DOWN","DOWN":"RIGHT","RIGHT":"UP"}[d]
    def _right_of(self,d): return {"UP":"RIGHT","RIGHT":"DOWN","DOWN":"LEFT","LEFT":"UP"}[d]

    # -------- state → 13‑d vector ----------
    def flatten(self,s):
        hx, hy          = s["snake"][0]
        fx, fy          = s["food"]
        W, H            = s["board_width"], s["board_height"]
        dir_            = s["direction"]
        body            = set(s["snake"])
        nxt = {
            "UP":    (hx, hy-1),
            "DOWN":  (hx, hy+1),
            "LEFT":  (hx-1, hy),
            "RIGHT": (hx+1, hy),
        }
        def danger(vec):
            x,y = nxt[vec]; wall = x<0 or x>=W or y<0 or y>=H
            return 1.0 if wall or (x,y) in body else 0.0
        d_left, d_fwd, d_right = danger(self._left_of(dir_)), danger(dir_), danger(self._right_of(dir_))
        dir_one_hot = [1 if d==dir_ else 0 for d in self.DIRECTIONS]
        food_flags  = [int(fy<hy), int(fy>hy), int(fx<hx), int(fx>hx)]
        return np.array([d_left,d_fwd,d_right,*dir_one_hot,*food_flags],dtype=np.float32)

    # -------- tick callbacks (called by engine) ----------
    def note_tick(self, state, prev_dist):
        self.step_states.append(self.flatten(state))
        self.step_actions.append(self.last_action)

        # base tick penalty
        r = PER_TICK_PENALTY

        # distance shaping
        hx,hy = state["snake"][0]
        fx,fy = state["food"]
        new_dist = manhattan(hx,hy,fx,fy)
        if new_dist < prev_dist:   r +=  DIST_BONUS
        elif new_dist > prev_dist: r += -DIST_BONUS

        self.step_rewards.append(r)

    def note_food(self):
        self.step_rewards[-1] += FOOD_REWARD     # bump last action

    def note_death(self):
        if not self.step_rewards: self.step_rewards.append(0.0)
        self.step_rewards[-1] += -1.0

    # -------- pick action ----------
    def act(self,state):
        x = self.flatten(state)
        if random.random() < self.epsilon:
            a_idx = random.randint(0,3)
        else:
            logits = self.model(tf.convert_to_tensor([x]))[0].numpy()
            a_idx  = int(np.argmax(logits))
        self.last_action = a_idx
        return self.IDX_TO_DIRECTION[a_idx]

    # -------- learning ----------
    def finish_episode(self):
        if not self.step_rewards: return
        R, returns = 0, deque()
        for r in reversed(self.step_rewards):
            R = r + GAMMA*R
            returns.appendleft(R)
        returns = np.array(returns,dtype=np.float32)

        states  = np.stack(self.step_states)
        acts    = np.array(self.step_actions)

        with tf.GradientTape() as tape:
            logits   = self.model(states, training=True)
            logpi    = tf.nn.log_softmax(logits)
            sel_log  = tf.reduce_sum(tf.one_hot(acts,4)*logpi,axis=1)
            loss     = -tf.reduce_mean(sel_log * returns)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

        # stats / decay
        self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)
        self.ep += 1
        self.avg_return += (returns[0]-self.avg_return)/self.ep
        if self.ep % 25 == 0:
            print(f"[Ep {self.ep}] avgR={self.avg_return:.3f}  ε={self.epsilon:.3f}")
        self.ret_history.append(self.avg_return)
        self.reset_ep()

    def reset_ep(self):
        self.step_states, self.step_actions, self.step_rewards = [],[],[]
        self.last_action = 0

# ----------------------------- main -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=int, nargs='?', default=1000)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--load", action="store_true")
    args = parser.parse_args()

    import snake_game
    bot = SnakeBot()

    if args.load and pathlib.Path("snake_policy.h5").exists():
        bot.model = tf.keras.models.load_model("snake_policy.h5")
        print("Loaded weights from snake_policy.h5")

    for ep in range(1, args.episodes+1):
        # --- run one episode ---
        game = snake_game.SnakeGame(snake_game.BOARD_W, snake_game.BOARD_H)
        prev_dist = manhattan(*game.snake[0], *game.food)

        while game.alive and game.ticks < snake_game.MAX_TICKS:
            state = {
                "board_width":game.w,"board_height":game.h,"snake":tuple(game.snake),
                "food":game.food,"direction":game.direction,
                "tick":game.ticks,"score":game.score
            }
            move = bot.act(state)
            game.step(move)
            bot.note_tick(state, prev_dist)
            prev_dist = manhattan(*game.snake[0], *game.food)
            if game.score > state["score"]: bot.note_food()
            game.ticks += 1
        bot.note_death()
        bot.finish_episode()

        if ep % 10 == 0:
            print(f"Ep {ep:4d} | Score={game.score:2d}  Ticks={game.ticks:4d}  "
                  f"Comp={game.ticks+10*game.score:4d}")

    if args.save:
        bot.model.save("snake_policy.h5")
        print("Weights saved to snake_policy.h5")

    # --- plot avgR curve ---
    plt.plot(range(1,len(bot.ret_history)+1), bot.ret_history)
    plt.xlabel("Episode"); plt.ylabel("avgR")
    plt.title("Learning curve"); plt.show()
