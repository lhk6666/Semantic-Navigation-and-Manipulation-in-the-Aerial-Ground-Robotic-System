import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from openai import OpenAI
from gpt_prompt import prompt_engineering_ as pe
from pydantic import BaseModel
from typing import List

class Step(BaseModel):
    explanation: str
    output: str

class MathReasoning(BaseModel):
    steps: List[Step]
    final_answer: str

class GPTReasoner:
    def __init__(self):
        self.openai = OpenAI()
        self.common_model = "gpt-4o"
        self.reasoning_model = "o1-mini"
        self.global_map = ''
        
    def global_planner(self, message_t):
        message = pe.prompt_glo_pln
        message.append({"role":"user", "content":f"Please decompose the task {message_t}. The format of your output should only include the contains shown in the following format: [[robot_name]: 'Task1'; [robot_name]: 'Task2'; [robot_name]: 'Task3'; ...]."})
        decision = self.openai.chat.completions.create(
                    model=self.common_model,
                    messages= message,
                    temperature = 0.1
                )
        decision_text = decision.choices[0].message.content
        return decision_text
    
    def mf_planner(self, command):
        message_loc = []
        namespace = {}
        exec('message_loc = pe.' + pe.prompt_assistant(command).strip('[]'), globals(), namespace)
        message_loc = namespace.get('message_loc')
        message_loc.append({"role": "user", "content": f"The task is {command}.\nPlease generate the motion function(s) to finish the sub-task."})
        decision = self.openai.chat.completions.create(
                    model=self.common_model,
                    messages= message_loc,
                    temperature = 0.1
                )
        decision_text = decision.choices[0].message.content
        return decision_text
    
    def map_constructor(self, component):
        input = []
        input.append(
            {
                "role": "user",
                "content": component + "\nPlease construct the global map based on all available local map data. There is only one instance of each object type. Only calculate the positions for cubes (or letters) and the robot.\n\n\
                    Important:\n\
                    1. Some local maps may contain incorrect information (e.g., a cube might be mislabeled).\n\
                    2. Analyze all data comprehensively rather than relying solely on individual local map calculations.\n\
                    3. If two objects are positioned unrealistically close (for example, two cubes appearing only 1.5 units apart when a cube’s center is defined by a side length of over 2.5 units), you must determine which local map contains the labeling error and correct it. Do not average or ignore the discrepancy—use the global consistency to infer the proper object identification and position.\n\
                    4. Global conditions remain constant. If an object appears in multiple local maps with conflicting labels (for instance, 'L' in one map and 'T' in another where the correct label should be 'L'), identify the erroneous local map and update the global map accordingly.\n\nOutput only the global map with each object’s name and coordinates. Do not include any additional calculations, explanations, or extra information."
            }

        )
        decision = self.openai.chat.completions.create(
                    model=self.reasoning_model,
                    messages= input,
                )
        self.global_map = decision.choices[0].message.content
        return self.global_map
            
    def map_updater(self, response, cord):
        local_info = 'When the world coordinates of the central of image is: ' + cord + '\nThe local coordinate of objects in the image is:\n' + response + '\n'
        update = self.openai.chat.completions.create(
                        model=self.reasoning_model,
                        messages=[
                                    {
                                        "role": "user",
                                        "content": "You will be given an original global map of the world. Please update this global map using the new local map data. Do not perform any averaging or additional calculations; simply update the global map with the new information. Output only the updated global map, including each object's name and its coordinates.\n\n\
                                                    Important:\n\
                                                    1. The local map may contain errors, such as misidentified objects. Even if an object is misidentified, analyze the error and deduce the correct identification based on letter similarity (since the original letter blocks in the global map do not change).\n\
                                                    2. We will not add new objects to the global map. If a new object appears in the local map, consider that it may be a misidentification (for example, L, T, and I are easily confused, as are V, A, and U) and update the global map accordingly."
                                    },
                                    {
                                        "role": "user",
                                        "content": "This is the original global map:\n" + self.global_map + "\nThis is the newest local map information:\n" + local_info + "\nPlease update the global map with this new local map and output the updated global map."
                                    }
                                ]
                    )
        self.global_map = update.choices[0].message.content
        return self.global_map, local_info

    def map_getter(self):
        return self.global_map      
        
if __name__ == '__main__':
    import time
    gpt_reasoner = GPTReasoner()
    start_time = time.time()
    # gpt_reasoner.global_planner("Move the red block to the right of the green block.")
    # gpt_reasoner.mf_planner("Move the red block to the right of the green block.")
    text = 'When the world coordinates of the central of image is: (4.0,4.0)\n\
            The local coordinate of objects in the image is:\n\
            name: Block_L, coordinate: x: -0.5, y: -0.5\n\
            name: Block_X, coordinate: x: -11.5, y": 4.8\n\
            name: Robot_Dog, coordinate: x: -6.0, y: -2.5\n\n\
            When the world coordinates of the central of image is: (-3.5,3.5)\n\
            The local coordinate of objects in the image is:\n\
            name: Block_I, coordinate: x: 10.5, y: -6.5\n\
            name: Block_L, coordinate: x: 8.4, y: 0.6\n\n\
            name: Block_U, coordinate: x: -5.5, y: -5.3\n\
            name: Robot_Dog, coordinate: x: 3.2, y: -1.6\n\n\
            When the world coordinates of the central of image is: (-3.5,-4.0)\n\
            The local coordinate of objects in the image is:\n\
            name: Block_U, coordinate: x: -6.0, y: 3.0\n\
            name: Block_I, coordinate: x: 11.0, y: 2.7\n\
            name: Tool_Box, coordinate: x: 5.0, y: -6.0\n\n\
            When the world coordinates of the central of image is: (4.0,-4.0)\n\
            The local coordinate of objects in the image is:\n\
            name: Block_I, coordinate: x: 2.8, y: 2.5\n\
            name: Red_Wire, coordinate: x: 0.8, y: -4.0\n\
            name: Tool_Box, coordinate: x: -7.6, y: -5.4\n\
            name: Robot_Dog, coordinate: x: -7.2, y: -6.0\n\n\
            When the world coordinates of the central of image is: (0.0,0.0)\n\
            The local coordinate of objects in the image is:\n\
            name: Block_I, coordinate: x: 7.5, y: -2.2\n\
            name: Block_U, coordinate: x: -9.6, y: -2.5\n\
            name: Block_X, coordinate: x: 3.6, y: 4.8\n\
            name: Robot_Dog, coordinate: x: -2.0, y: 2.7\n\n'
    text_ = gpt_reasoner.map_constructor(text)
    print(text_, time.time() - start_time)
    text = 'When the world coordinates of the central of image is: -6.5, 5.0\n\
            The local coordinate of objects in the image is:\n\
            name: Block_V, coordinate: x: -0.2, y: 0.8\n\
            name: Robot_Dog, coordinate: x: 0.5, y: -2.9\n\n'
    start_time = time.time()
    text_,_ = gpt_reasoner.map_updater(text, "-6.5, 5.0")
    print(text_, time.time() - start_time)
    # gpt_reasoner.map_updater("Move the red block to the right of the green block.", "The red block is on the left of the green block.", "0, 0")