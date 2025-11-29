#!/usr/bin/env python3
import sys
import os
import rospy
import json
import numpy as np
from motions import DatasetMotions
from std_msgs.msg import String

def evaluate():
    rospy.init_node('dataset_evaluator')
    
    dataset_path = "/media/dragon_llm/3C09549315E08290/vla_dataset_semantic_ex"
    
    # Initialize DatasetMotions
    # We need to pass parameters that match the dataset image specs if possible
    # Dataset image is 640x480.
    # DogMotions default resolution is 1080 (1920x1080).
    # Our mock_perception handles the scaling, so we can keep defaults or adjust.
    
    motions = DatasetMotions(dataset_path=dataset_path)
    
    # Get list of samples
    # We can list files in the dataset path
    # Since we can't list directly in python if it's restricted, we might need to rely on sequential IDs
    # The user showed sample_000000.json, sample_000001.json...
    
    num_samples = 10 # Start with a small number for testing
    success_count = 0
    
    for i in range(num_samples):
        if rospy.is_shutdown():
            break
            
        rospy.loginfo(f"Evaluating Sample {i}")
        
        # Load Sample
        if not motions.load_sample(i):
            rospy.logwarn(f"Skipping sample {i}")
            continue
            
        # Get Instruction
        instruction = motions.current_sample['instruction']
        target_object = motions.current_sample['metadata']['target_object']
        
        rospy.loginfo(f"Instruction: {instruction}, Target: {target_object}")
        
        # Execute Task
        # We assume the task is always "navigate to X"
        # So we call go1_following_start(target_object)
        
        try:
            motions.go1_following_start(target_object)
            
            # Check Success
            # Success criteria: Robot is close to target
            # DatasetMotions updates robot_position.
            # We can check distance to target.
            
            # Re-calculate target pos in local frame
            scale_x = 1920 / 80.0
            scale_y = 1080 / 80.0
            goal_norm = motions.current_sample['metadata']['goal_position']
            target_x = (goal_norm[0] - 0.5) * scale_x
            target_y = (0.5 - goal_norm[1]) * scale_y
            
            dist = np.linalg.norm(motions.robot_position - np.array([target_x, target_y]))
            rospy.loginfo(f"Final Distance: {dist}")
            
            if dist < 2.0: # Threshold (approx 2 units?)
                success_count += 1
                rospy.loginfo("Success!")
            else:
                rospy.loginfo("Failure.")
                
        except Exception as e:
            rospy.logerr(f"Error executing sample {i}: {e}")
            
    rospy.loginfo(f"Evaluation Complete. Success Rate: {success_count}/{num_samples}")

if __name__ == '__main__':
    evaluate()
