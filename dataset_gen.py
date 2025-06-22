import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import numpy as np
import cv2
from tqdm import tqdm
import os

class MarioFrameDatasetGenerator:
    def __init__(self, output_dir="mario_frame_dataset", num_episodes=100, frames_per_episode=1000):
        try:
            self.env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
            self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        except Exception as e:
            print(f"Error initializing Mario environment: {e}")
            raise
            
        self.output_dir = output_dir
        self.num_episodes = num_episodes
        self.frames_per_episode = frames_per_episode
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/current_frames", exist_ok=True)
        os.makedirs(f"{output_dir}/next_frames", exist_ok=True)
        
    def preprocess_frame(self, frame):
        # Convert to grayscale and resize to 64x64
        frame = cv2.resize(frame, (128, 128))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame
    
    def generate_dataset(self):
        frame_pairs = []
        actions = []
        frame_count = 0
        
        for episode in tqdm(range(self.num_episodes)):
            current_frame = self.env.reset()
            
            for step in range(self.frames_per_episode):
                # Take random action
                action = self.env.action_space.sample()
                next_frame, reward, done, info = self.env.step(action)
                
                # Preprocess frames
                current_frame_proc = self.preprocess_frame(current_frame)
                next_frame_proc = self.preprocess_frame(next_frame)
                
                # Save frames
                cv2.imwrite(f"{self.output_dir}/current_frames/{frame_count}.png", current_frame_proc)
                cv2.imwrite(f"{self.output_dir}/next_frames/{frame_count}.png", next_frame_proc)
                
                # Store action
                actions.append(action)
                frame_count += 1
                
                current_frame = next_frame
                
                if done:
                    break
        
        # Save actions
        np.save(f"{self.output_dir}/actions.npy", np.array(actions))
        
    def __del__(self):
        if hasattr(self, 'env'):
            self.env.close()


# Usage
if __name__ == "__main__":
    generator = MarioFrameDatasetGenerator(
        output_dir="mario_frame_dataset",
        num_episodes=10,
        frames_per_episode=1000
    )
    generator.generate_dataset()
