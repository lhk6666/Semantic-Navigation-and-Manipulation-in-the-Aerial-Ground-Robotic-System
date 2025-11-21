
# llm_multiagent

This project is part of the **LLM-based robot system**, focusing on hierarchical language model coordination for aerial-ground robotic systems. The system enables a UAV and a quadruped ground robot to cooperatively perform semantic navigation and object manipulation using language models.

> **Core paper**: _Hierarchical Language Models for Semantic Navigation and Manipulation in Aerial-Ground Robotic System_  
> _**Started**: October 2024_

---

## 🧠 Overview

This package implements a hierarchical LLM control system for a multi-agent robot team, consisting of:

- A **quadrotor** running SLAM and aerial perception.
- A **Unitree Go1** quadruped robot performing manipulation tasks.
- A **hierarchical multi-agent LLM system** that interprets high-level instructions and coordinates the agents.

Tasks include object relocation (e.g., "move the blue cube to the right side of the red cube") and spatial arrangements (e.g., "arrange the color cubes red, blue, green from left to right"). Reasoning-based constraints are also supported, such as fixing one block’s position.

---

## 🔑 Authentication for Gemini Model

This project uses the **Gemini** model hosted on **Google Cloud** for high-level language reasoning.  
To use it, please follow the login procedure described in this internal documentation (Temporarily: Dragon Lab members only):

📂 [Gemini Access Guide (Google Drive)](https://drive.google.com/drive/folders/1KVDLnAkX0T4J5hMR2z7AR7iq1y8wRyBP?usp=drive_link)

Make sure to:

1. Login using the provided account/password.
2. Follow the instructions in the `readme` inside the Drive folder for terminal-based authentication.

⚠️ Notice: While Dragonlab’s API is not publicly accessible, external researchers are welcome to download our dataset by fill in this [form](https://docs.google.com/forms/d/e/1FAIpQLSc9VKcZ8bRAFB7rwyjrG0NZAhwAAipH0I8pKLLlgG56Jg5mdA/viewform?usp=sharing&ouid=105560342723483080065) and send a email to the authors. This dataset can reproduce the model by fine-tuning Gemini.
The model is only fine-tuned with the real-world dataset with letter cubes. Based on the fine-tuning, the model can finish the simulated color cubes transportation in zero-shot without any fine-tuning.

---

## 🚀 Launch Instructions

### Simulation Mode

> ⚠️ The first command belongs to the [jsk_aerial_robot](https://github.com/lhk6666/jsk_aerial_robot) package.  
> Make sure to switch to [**Haokun LIU's fork**](https://github.com/lhk6666/jsk_aerial_robot) and select the **`Haokun` branch** before proceeding.

### Launch the simulation environmet (jsk_aerial_robot)
```bash
roslaunch go1_description bringup.launch real_machine:=false simulation:=True headless:=False
```

### Launch the LLM controller node in simulation mode (This)
```bash
roslaunch ros_chatgpt main.launch simulation:=true if_record_data:=false if_record_video:=false map_updater:=true
```

### Real-world Mode (This)

```bash
roslaunch ros_chatgpt main.launch
```

---

## 🧩 Supported Commands

This system currently supports the following types of natural language instructions:

- **Object relocation**:  
  Example:
  `move the blue cube to the right side of the red cube`, `move the green cube to the (0, 0)`

- **Directional references**:  
  `left`, `right`, `back`, `front`

- **Sequencing and arrangement**:  
  Example: `arrange the color cubes with sequence red, blue, green from left to right`

- **Reasoning constraints**:  
  You can impose logical restrictions like `"do not move the red cube"` to evaluate the LLM’s reasoning ability.

---

## 📂 Folder Structure

```
llm_multiagent/
├── README.md                 # You're here
├── main.py                   # Allocate and execute sub-tasks
├── constructor.py            # Call planners and reasoners to analysis local map and construct global map
├── launch/                   # Launch files
├── reasoners/                # Reasoners include chatgpt-based and gemini-based (used)
├── planner/                  # Planner include the gemini-based local map planner and the planners to      
                                generate drone and robot dog's motions
└── ...
```

---

## 📬 Contact

For technical questions or collaboration inquiries, please contact the Dragon Lab development team. (Email: haokun-liu@dragon.t.u-tokyo.ac.jp)