# bot.py  –  full upgraded SnakeBot (all tweaks)
import tensorflow as tf 
import numpy as np
import random, argparse, pathlib, sys

# optional plotting
try:
    import matplotlib.pyplot as plt
    PLOT = True
except ModuleNotFoundError:
    PLOT = False

from collections import deque

# ───────── hyper‑params ──────────
GAMMA             = 0.99 #discount parameter for future rewards
PER_TICK_PENALTY  = -0.0025
FOOD_REWARD       =  5.0
DIST_BONUS        =  0.25       # + if closer, – if farther
ENTROPY_COEF      =  0.001

# ───────── helper ──────────
def manhattan(ax, ay, bx, by): return abs(ax-bx)+abs(ay-by)

# ───────── SnakeBot class ─────────
class SnakeBot:
    DIRECTIONS        = ["UP","DOWN","LEFT","RIGHT"]
    IDX_TO_DIRECTION  = dict(enumerate(DIRECTIONS))
    DIRECTION_TO_IDX  = {d:i for i,d in IDX_TO_DIRECTION.items()}

    def __init__(self):
        self.score_history = []
        self.model = self._build_model()
        self.opt   = tf.keras.optimizers.Adam(1e-3)

        self.epsilon       = 0.20
        self.eps_min       = 0.01
        self.eps_decay     = 0.995

        self.reset_ep()
        self.ep            = 0
        self.avg_return    = 0
        self.ret_history   = []
        self.best_score    = 0

    # -------- network ----------
    def _build_model(self):
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=(11,)),
            tf.keras.layers.Dense(128), tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(64),  tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(4)   # logits for 4 moves
        ])

    # -------- geometry helpers ----------
    def _left_of(self,d):  return {"UP":"LEFT","LEFT":"DOWN","DOWN":"RIGHT","RIGHT":"UP"}[d]
    def _right_of(self,d): return {"UP":"RIGHT","RIGHT":"DOWN","DOWN":"LEFT","LEFT":"UP"}[d]

    # -------- state → 11‑d features ----------
    def flatten(self,s):
        hx, hy  = s["snake"][0]
        fx, fy  = s["food"]
        W, H    = s["board_width"], s["board_height"]
        d       = s["direction"]
        body    = set(s["snake"])

        nxt = {"UP":(hx,hy-1),"DOWN":(hx,hy+1),"LEFT":(hx-1,hy),"RIGHT":(hx+1,hy)}
        def danger(vec):
            x,y = nxt[vec]; return 1.0 if x<0 or x>=W or y<0 or y>=H or (x,y) in body else 0.0
        dL, dF, dR = danger(self._left_of(d)), danger(d), danger(self._right_of(d))

        dir_one_hot = [1 if d==k else 0 for k in self.DIRECTIONS]
        food_flags  = [int(fy<hy), int(fy>hy), int(fx<hx), int(fx>hx)]
        return np.array([dL,dF,dR,*dir_one_hot,*food_flags], dtype=np.float32)

    # -------- tick callbacks ----------
    def note_tick(self, state, prev_dist):
        self.step_states.append(self.flatten(state))
        self.step_actions.append(self.last_action)

        r = PER_TICK_PENALTY
        hx,hy = state["snake"][0]; fx,fy = state["food"]
        dist  = manhattan(hx,hy,fx,fy)
        if   dist < prev_dist: r +=  DIST_BONUS
        elif dist > prev_dist: r += -DIST_BONUS
        self.step_rewards.append(r)

    def note_food(self):
        self.step_rewards[-1] += FOOD_REWARD

    def note_death(self):
        if not self.step_rewards: self.step_rewards.append(0.0)
        self.step_rewards[-1] += -3.0

    # -------- choose action ----------
    def act(self, state):
        x = self.flatten(state)
        if random.random() < self.epsilon:
            a_idx = random.randint(0,3)
        else:
            logits = self.model(tf.convert_to_tensor([x]))[0].numpy()
            a_idx  = int(np.argmax(logits))
        self.last_action = a_idx
        return self.IDX_TO_DIRECTION[a_idx]

    # -------- learning update ----------
    def finish_episode(self):
        if not self.step_rewards: return
        # discounted returns
        R, returns = 0, deque()
        for r in reversed(self.step_rewards):
            R = r + GAMMA * R
            returns.appendleft(R)
        returns = np.array(returns, dtype=np.float32)
        returns -= returns.mean()                # advantage normalization

        states = np.stack(self.step_states)
        acts   = np.array(self.step_actions)

        with tf.GradientTape() as tape:
            logits  = self.model(states, training=True)
            logpi   = tf.nn.log_softmax(logits)
            sel_log = tf.reduce_sum(tf.one_hot(acts,4) * logpi, axis=1)
            entropy = tf.reduce_sum(tf.exp(logpi) * -logpi, axis=1)

            loss = -tf.reduce_mean(sel_log * returns + ENTROPY_COEF * entropy)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

        # stats / ε‑decay
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)
        self.ep += 1
        self.avg_return += (returns[0] - self.avg_return) / self.ep
        if self.ep % 25 == 0:
            print(f"[Ep {self.ep}] avgR={self.avg_return:.3f}  ε={self.epsilon:.3f}")
        self.ret_history.append(self.avg_return)
        self.reset_ep()

    def reset_ep(self):
        self.step_states, self.step_actions, self.step_rewards = [], [], []
        self.last_action = 0

# ──────────── main training script ────────────
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

    for ep in range(1, args.episodes + 1):
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
            if game.score > state["score"]:
                bot.note_food()
            game.ticks += 1

        bot.note_death()
        # track best score (food count)
        if game.score > bot.best_score:
            bot.best_score = game.score
        bot.finish_episode()
        bot.score_history.append(game.score)

        # --- prints ---
        if ep % 10 == 0:
            print(f"Ep {ep:4d} | Score={game.score:2d}  "
                  f"Ticks={game.ticks:4d}  "
                  f"Comp={game.ticks + 10*game.score:4d}")

        # --- checkpoint every 500 ---
        if args.save and ep % 500 == 0:
            fname = f"chkpt_{ep}.h5"
            bot.model.save(fname)
            print(f"Checkpoint saved → {fname}")
            print(f"Best score so far: {bot.best_score} food")

    # save final weights
    if args.save:
        bot.model.save("snake_policy.h5")
        print("Weights saved to snake_policy.h5")

    # plot learning curve
    if PLOT:
        fig, ax1 = plt.subplots()
        ax1.plot(range(1, len(bot.ret_history)+1), bot.ret_history, label="avgR", color="tab:blue")
        ax1.set_xlabel("Episode"); ax1.set_ylabel("avgR", color="tab:blue")
        ax1.tick_params(axis='y', labelcolor="tab:blue")

        ax2 = ax1.twinx()
        ax2.plot(range(1, len(bot.score_history)+1), bot.score_history,
                label="Score", color="tab:orange", alpha=0.5)
        ax2.set_ylabel("Food per episode", color="tab:orange")
        ax2.tick_params(axis='y', labelcolor="tab:orange")

        plt.title("Learning curve")
        fig.tight_layout()
        plt.show()
