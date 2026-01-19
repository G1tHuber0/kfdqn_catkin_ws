# HYAS：监督/探索阶段直接用模糊动作；否则 softmax 融合
q_values = self.q_net(state)
fuzzy_logits = self.fuzzy_guide(state)
a_f = int(fuzzy_logits.argmax(dim=1).item())
if (np.random.rand() <= self.epsilon) or (episode_idx <= self.cfg.ep_r):
    return a_f, 'a_f', None
q_score = F.softmax(q_values, dim=1)
f_score = F.softmax(fuzzy_logits, dim=1)
hybrid_score = self.h1 * f_score + self.h2 * q_score
hya = int(hybrid_score.argmax(dim=1).item())
a_q = int(q_values.argmax(dim=1).item())
return hya, 'hya', a_q

# HYL：混合 TD 目标（m * max_next + n * Q(s', a_f)）
q_sa = self.q_net(states).gather(1, actions)
with torch.no_grad():
    max_next = self.target_q_net(next_states).max(dim=1)[0].view(-1, 1)
    a_f_next = self.fuzzy_guide(next_states).argmax(dim=1).view(-1, 1)
    q_fuzzy_next = self.q_net(next_states).gather(1, a_f_next)
    hybrid_next = self.m * max_next + self.n * q_fuzzy_next
    q_target = rewards + self.cfg.gamma * hybrid_next * (1.0 - dones)

q_loss = F.mse_loss(q_sa, q_target)


CHECKPOINT_STEPS = [2000, 5000, 10000, 20000, 30000, 50000, 75000, 100000, 150000]

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"{ALGO_NAME}_seed{cfg.seed}_{timestamp}"
output_dir = os.path.join(BASE_DIR, "outputs", OUTPUT_ENV_DIR, run_name)
log_dir = os.path.join(output_dir, "logs")
model_dir = os.path.join(output_dir, "models")
data_dir = os.path.join(output_dir, "data")

os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

writer = SummaryWriter(log_dir=log_dir)

# 训练 loop 内（节选）
if total_steps % 10 == 0:
    writer.add_scalar("Step/01_Loss", current_loss, total_steps)
    if ALGO_NAME == "KFDQN":
        writer.add_scalar("Step/02_Q_Loss", q_loss_val, total_steps)
        writer.add_scalar("Step/03_Fuzzy_Loss", fuzzy_loss_val, total_steps)

if total_steps in CHECKPOINT_STEPS:
    save_name = f"{ALGO_NAME}_{timestamp}_{total_steps}.pth"
    save_path = os.path.join(model_dir, save_name)
    agent.save(save_path)
    tqdm.write(f">>> [Checkpoint] Model saved: {save_name} at step {total_steps}")


    # __init__ 里（节选）
self.goal_file = goal_file          # 新增：保存文件名
self.goal_list_path = pathlib.Path(__file__).resolve().parent / self.goal_file
print(f"[*] ROS Environment: Loading goal list from {self.goal_list_path}")
self._goal_list = None  # 保持延迟加载逻辑

def _load_goal_list_if_needed(self):
    if self._goal_list is not None:
        return
    if not self.goal_list_path.exists():
        raise FileNotFoundError(
            f"Fixed goal list not found: {self.goal_list_path}. "
            f"Please generate it first (tools/gen_goal_list.py)."
        )

    goals = []
    with self.goal_list_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gx = float(row["goal_x"])
            gy = float(row["goal_y"])
            goals.append((gx, gy))

    if len(goals) == 0:
        raise ValueError(f"Empty goal list: {self.goal_list_path}")

    self._goal_list = goals

def _get_fixed_goal(self, episode_idx: int):
    self._load_goal_list_if_needed()
    assert self._goal_list is not None
    i = episode_idx % len(self._goal_list)
    return self._goal_list[i]
