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
    def __init__(self, eval_env, render_freq=1000, verbose=0):
        super(VideoRecorderCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.render_freq = render_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            screens = []
            obs, _ = self.eval_env.reset()
            for _ in range(100):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = self.eval_env.step(action)
                screen = self.eval_env.render() # (H, W, 3)
                screens.append(screen)
                if terminated or truncated:
                    break
            
            if len(screens) > 0:
                # Convert list to (T, H, W, C) for TensorBoard
                video_array = np.array(screens)
                video_array = video_array.transpose(0, 3, 1, 2) # (T, C, H, W)
                video_array = np.expand_dims(video_array, axis=0) # (1, T, C, H, W)
                
                # Log to TensorBoard
                self.logger.record("trajectory/video", Video(video_array, fps=5), exclude=("stdout", "log", "json", "csv"))
        return True

# Helper to wrap SB3 logger for Video
from stable_baselines3.common.logger import Video

def train():
    # 0. Path Setup
    base_path = "./Stage-2-MLP-MiniGrid/"
    log_path = os.path.join(base_path, "logs/")
    model_path = os.path.join(base_path, "models/")
    video_path = os.path.join(base_path, "videos/")
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(video_path, exist_ok=True)

    # 1. Environment Setup
    # MiniGrid-Empty-8x8-v0 is a simple 8x8 grid where the agent needs to reach the green square.
    env_id = "MiniGrid-Empty-8x8-v0"
    
    # We use FlatObsWrapper to convert the dict observation (image, direction, etc.) 
    # into a single flat vector that a standard Multi-Layer Perceptron (MLP) can process.
    env = gym.make(env_id, render_mode="rgb_array")
    env = FlatObsWrapper(env)
    env = Monitor(env) # For logging

    # 2. Model Definition
    # We use PPO (Proximal Policy Optimization), a robust and efficient RL algorithm.
    # Smallest MLP structure: 2 hidden layers with 16 neurons each.
    # policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
    policy_kwargs = dict(net_arch=dict(pi=[128, 64], vf=[128, 64]))
    learning_rate = 1e-3
    n_steps = 2048
    ent_coef = 0.01

    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        policy_kwargs=policy_kwargs,
        tensorboard_log=os.path.join(log_path, "ppo_minigrid_tensorboard/"),
        learning_rate=learning_rate, # Increased learning rate
        n_steps=n_steps,        # Larger rollout for more stability
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=ent_coef,      # Encourage exploration
    )

    # 3. Training
    print(f"Starting training on {env_id}...")
    
    # Callbacks for saving and evaluation
    eval_env = gym.make(env_id, render_mode="rgb_array")
    eval_env = FlatObsWrapper(eval_env)
    
    # Wrap eval_env to record videos
    # We will record every evaluation to combine them later
    temp_video_folder = os.path.join(video_path, "temp_evals/")
    os.makedirs(temp_video_folder, exist_ok=True)
    eval_env = RecordVideo(eval_env, video_folder=temp_video_folder, 
                           episode_trigger=lambda x: True,
                           name_prefix="eval")

    eval_callback = EvalCallback(eval_env, best_model_save_path=os.path.join(model_path, "best_model/"),
                             log_path=log_path, eval_freq=2000, 
                             n_eval_episodes=1, # Only 1 episode per eval to keep video concise
                             deterministic=True, render=False)
    
    total_timesteps = 80000 
    model.learn(total_timesteps=total_timesteps, callback=[eval_callback])
    
    # 4. Save the model
    model_save_file = os.path.join(model_path, "ppo_minigrid_small_mlp_stage2")
    model.save(model_save_file)
    print(f"Model saved to {model_save_file}")

    # 5. Combine videos into one and Speed Up
    print("Combining evaluation videos and speeding up to fit 60s...")
    try:
        video_files = sorted([os.path.join(temp_video_folder, f) for f in os.listdir(temp_video_folder) if f.endswith(".mp4")])
        if video_files:
            clips = [VideoFileClip(f) for f in video_files]
            combined_clip = concatenate_videoclips(clips)
            
            # Target duration: 50 seconds for training part to leave room for testing part in final summary
            target_duration = 50.0
            if combined_clip.duration > target_duration:
                print(f"Original duration: {combined_clip.duration:.2f}s. Speeding up to {target_duration}s.")
                final_training_clip = combined_clip.with_speed_scaled(final_duration=target_duration)
            else:
                final_training_clip = combined_clip

            final_training_clip.write_videofile(os.path.join(video_path, "training_process.mp4"), fps=24, logger=None)
            
            # Close clips
            for clip in clips:
                clip.close()
            final_training_clip.close()
            print("Training video saved and sped up.")
    except Exception as e:
        print(f"Error combining/speeding up videos: {e}")

    # 6. Testing/Evaluation
    print("\nEvaluating the trained model...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # 7. Generate results.md
    results_path = os.path.join(base_path, "results.md")
    with open(results_path, "w") as f:
        f.write(f"# Stage 2 Results - {env_id}\n\n")
        f.write(f"## Model Performance\n")
        f.write(f"- **Mean Reward:** {mean_reward:.2f}\n")
        f.write(f"- **Standard Deviation:** {std_reward:.2f}\n")
        f.write(f"- **Total Timesteps:** {total_timesteps}\n\n")
        f.write(f"## Configuration\n")
        f.write(f"- **Algorithm:** PPO\n")
        f.write(f"- **Policy:** MlpPolicy\n")
        f.write(f"- **Architecture:** {policy_kwargs['net_arch']}\n")
        f.write(f"- **Learning Rate:** {learning_rate}\n")
        f.write(f"- **Ent Coef:** {ent_coef}\n")
        f.write(f"- **N Steps:** {n_steps}\n\n")
        f.write(f"## Artifacts\n")
        f.write(f"- **Model Path:** `{model_save_file}`\n")
        f.write(f"- **Video Path:** `{os.path.join(video_path, 'training_summary.mp4')}`\n")
    print(f"Results summary saved to {results_path}")

def test():
    # 0. Path Setup
    base_path = "./Stage-2-MLP-MiniGrid/"
    model_path = os.path.join(base_path, "models/")
    video_path = os.path.join(base_path, "videos/")

    # Load and play
    env_id = "MiniGrid-Empty-8x8-v0"
    print(f"Loading environment {env_id} for video testing...")
    
    # Use 'rgb_array' to record video
    env = gym.make(env_id, render_mode="rgb_array") 
    env = FlatObsWrapper(env)
    
    # Record a single video of the testing phase
    env = RecordVideo(env, video_folder=video_path, 
                     episode_trigger=lambda x: True,
                     name_prefix="testing_final")
    
    try:
        model_load_file = os.path.join(model_path, "ppo_minigrid_small_mlp_stage2")
        model = PPO.load(model_load_file)
        obs, _ = env.reset()
        
        total_reward = 0
        for i in range(100):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                print(f"\nEpisode finished in {i+1} steps | Total Reward: {total_reward:.2f}")
                break
        
        env.close() # Ensure video is saved

        # Combine training and testing video
        print("Creating final consolidated video...")
        try:
            final_video_path = os.path.join(video_path, "training_summary.mp4")
            training_video = os.path.join(video_path, "training_process.mp4")
            
            # Find the testing video
            test_files = sorted([os.path.join(video_path, f) for f in os.listdir(video_path) 
                            if f.startswith("testing_final") and f.endswith(".mp4")])
            
            if os.path.exists(training_video) and test_files:
                train_clip = VideoFileClip(training_video)
                test_clip = VideoFileClip(test_files[0])
                
                final_summary = concatenate_videoclips([train_clip, test_clip])
                final_summary.write_videofile(final_video_path)
                
                train_clip.close()
                test_clip.close()
                print(f"Final summary video saved to {final_video_path}")
        except Exception as e:
            print(f"Error creating final summary video: {e}")
                
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    # Create logs directory
    # os.makedirs("./logs/", exist_ok=True) # Now handled inside functions
    
    train()
    test() 
