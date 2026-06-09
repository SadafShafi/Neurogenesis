import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import Video
import os
import time
import numpy as np
from moviepy import VideoFileClip, concatenate_videoclips

class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env, render_freq=2000, verbose=0):
        super(VideoRecorderCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.render_freq = render_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            screens = []
            obs, _ = self.eval_env.reset()
            # 6x6 is small, but goal is random
            for _ in range(50):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = self.eval_env.step(action)
                screen = self.eval_env.render()
                screens.append(screen)
                if terminated or truncated:
                    break
            
            if len(screens) > 0:
                video_array = np.array(screens)
                video_array = video_array.transpose(0, 3, 1, 2)
                video_array = np.expand_dims(video_array, axis=0)
                self.logger.record("trajectory/video", Video(video_array, fps=5), exclude=("stdout", "log", "json", "csv"))
        return True

def train():
    # 0. Path Setup
    base_path = "./Stage-4-MLP-MiniGrid/"
    log_path = os.path.join(base_path, "logs/")
    model_path = os.path.join(base_path, "models/")
    video_path = os.path.join(base_path, "videos/")
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(video_path, exist_ok=True)

    # 1. Environment Setup
    # The goal and agent start positions are randomized in every episode
    env_id = "MiniGrid-Empty-Random-6x6-v0"
    env = gym.make(env_id, render_mode="rgb_array")
    env = FlatObsWrapper(env)
    env = Monitor(env)

    # 2. Model Definition
    # Keeping architecture same as Stage 3 [128, 64]
    policy_kwargs = dict(net_arch=dict(pi=[128, 64], vf=[128, 64]))
    learning_rate = 3e-4
    n_steps = 2048
    ent_coef = 0.01
    
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        policy_kwargs=policy_kwargs,
        tensorboard_log=os.path.join(log_path, "ppo_minigrid_tensorboard/"),
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=ent_coef,
    )

    # 3. Training
    print(f"Starting training on {env_id}...")
    
    eval_env = gym.make(env_id, render_mode="rgb_array")
    eval_env = FlatObsWrapper(eval_env)
    
    temp_video_folder = os.path.join(video_path, "temp_evals/")
    os.makedirs(temp_video_folder, exist_ok=True)
    eval_env = RecordVideo(eval_env, video_folder=temp_video_folder, 
                           episode_trigger=lambda x: True,
                           name_prefix="eval")

    eval_callback = EvalCallback(eval_env, best_model_save_path=os.path.join(model_path, "best_model/"),
                             log_path=log_path, eval_freq=2000, 
                             n_eval_episodes=1,
                             deterministic=True, render=False)
    
    total_timesteps = 60000 
    model.learn(total_timesteps=total_timesteps, callback=[eval_callback])
    
    # 4. Save the model
    model_save_file = os.path.join(model_path, "ppo_minigrid_stage4_random6x6")
    model.save(model_save_file)

    # 5. Video Processing
    print("Processing videos...")
    try:
        video_files = sorted([os.path.join(temp_video_folder, f) for f in os.listdir(temp_video_folder) if f.endswith(".mp4")])
        if video_files:
            clips = [VideoFileClip(f) for f in video_files]
            combined_clip = concatenate_videoclips(clips)
            target_duration = 50.0
            if combined_clip.duration > target_duration:
                final_training_clip = combined_clip.with_speed_scaled(final_duration=target_duration)
            else:
                final_training_clip = combined_clip
            final_training_clip.write_videofile(os.path.join(video_path, "training_process.mp4"), fps=24, logger=None)
            for clip in clips: clip.close()
            final_training_clip.close()
    except Exception as e: print(f"Video Error: {e}")

    # 6. Evaluation
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    
    # 7. Generate results.md
    results_path = os.path.join(base_path, "results.md")
    with open(results_path, "w") as f:
        f.write(f"# Stage 4 Results - {env_id}\n\n")
        f.write(f"## Model Performance\n")
        f.write(f"- **Mean Reward:** {mean_reward:.2f}\n")
        f.write(f"- **Standard Deviation:** {std_reward:.2f}\n")
        f.write(f"- **Total Timesteps:** {total_timesteps}\n\n")
        f.write(f"## Configuration\n")
        f.write(f"- **Architecture:** {policy_kwargs['net_arch']}\n")
        f.write(f"- **Learning Rate:** {learning_rate}\n")
        f.write(f"- **Ent Coef:** {ent_coef}\n")
        f.write(f"- **N Steps:** {n_steps}\n\n")
    print(f"Results summary saved to {results_path}")

def test():
    base_path = "./Stage-4-MLP-MiniGrid/"
    model_path = os.path.join(base_path, "models/")
    video_path = os.path.join(base_path, "videos/")
    env_id = "MiniGrid-Empty-Random-6x6-v0"
    
    env = gym.make(env_id, render_mode="rgb_array") 
    env = FlatObsWrapper(env)
    env = RecordVideo(env, video_folder=video_path, episode_trigger=lambda x: True, name_prefix="testing_final")
    
    try:
        model = PPO.load(os.path.join(model_path, "ppo_minigrid_stage4_random6x6"))
        obs, _ = env.reset()
        for i in range(50):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated: break
        env.close()
    except Exception as e: print(f"Test Error: {e}")

if __name__ == "__main__":
    train()
    test()
