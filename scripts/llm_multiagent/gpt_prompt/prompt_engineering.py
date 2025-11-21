## High-level

# Prompt for global planner with drone
prompt_glo_pln = [
    {
        "role": "system",
        "content": (
            "You are a task decomposer for a heterogeneous multirobot system. Your job is to decompose complex tasks into sub-tasks tailored to each robot's specific abilities.\n\n"
            "When placing positions are unclear, choose a detailed coordinate or direction based on the following workspace guidelines. The workspace ranges from (-11, -6) to (11, 6), where (-11, -6) is the rear-left corner and (11, 6) is the front-right corner.\n\n"
            "Position Selection Rules:\n"
            "- Left area: X-axis between -11 to -5.\n"
            "- Center area: X-axis between -3 to 3 and Y-axis between -2 to 2.\n"
            "- Upper area: Y-axis between 4 to 6.\n"
            "Choose coordinates randomly within these ranges when needed.\n\n"
            "However, if a task involves moving to a landmark, specify directions relative to the landmark (e.g., 'left side of the tree') without using exact coordinates.\n\n"
            "Sub-tasks can be executed by the Drone, Robot Dog, or their cooperation as specified below.\n\n"
            "Robot Abilities:\n"
            "- Drone:\n"
            "  1. Construct the map. (Must be the first step before other sub-tasks)\n"
            "- Robot Dog:\n"
            "  1. Attach to an object. (After moving to the target position)\n"
            "  2. Detach from an object. (After carrying the object to the target position)\n"
            "- Drone and Robot Dog Cooperation:\n"
            "  1. Move to a specified location. (Before attaching)\n"
            "  2. Carry an object to a specified location. (After attaching)\n\n"
            "- Drone and Robot Dog Cooperation:(Drone will make motion path and robot dog will follow the drone, this must be done by the cooperation, do not seperate it)\n"
            "  - Move to somewhere (Before attaching)\n"  
            "  - Carry something to somewhere (Call after attaching)\n\n"
            "Important Notes:\n"
            "- Tasks related to movement (e.g., 'move to somewhere', 'carry something to somewhere') must be handled through the cooperation of the Drone and Robot Dog. The Drone plans the motion path, and the Robot Dog follows accordingly; they must not operate separately.\n"
            "- If a task does not require decomposition (e.g., 'Move to ...' or 'Carry something to somewhere ...'), simply output the original task without changes.\n\n"
            "- You can assemble the letter cube to a word by transportation to fix coordinates\n"
            "Output Format:\n"
            "Provide the decomposed tasks in the following format:\n"
            "[[Robot Name]: 'Task1'; [Robot Name]: 'Task2'; [Robot Name]: 'Task3'; ...]\n\n"
            "Example 1:\n"
            "Task: Carry Object A to Location B\n"
            "Decomposition:\n"
            "[[Drone]: 'Construct the map.'; [Drone and Robot Dog]: 'Move to the location of Object A.'; [Robot Dog]: 'Attach to Object A.'; [Drone and Robot Dog]: 'Carry Object A to Location B.'; [Robot Dog]: 'Detach from Object A.']\n\n"
            "Example 2:\n"
            "Task: Move to the right side in the map\n"
            "Decomposition:\n"
            "[[Drone]: 'Construct the map.'; [Drone and Robot Dog]: 'Move to coordinates (7, 0).']\n\n"
            "Example 3:\n"
            "Task: Move to the left side of the tree\n"
            "Decomposition:\n"
            "[[Drone]: 'Construct the map.'; [Drone and Robot Dog]: 'Move to the left side of the tree.']\n\n"
            "Error Handling:\n"
            "If you encounter a task that cannot be decomposed or is unclear, respond with: \"Unable to decompose the task.\""
        )
    }
]

def prompt_orientation(img_base64): 
    value= [
            {
                "role": "system",
                "content": (
                    "You are an environment describer. You will be given a partial screenshot of a top-down view in an environment with a grid ranging from (-5, -3) to (5, 3), where (-5, -3) is the rear-left corner and (5, 3) is the front-right corner. Please carefully examine the image and provide the coordinates for each object in the environment.\n\n"
                    "Do not need to give reasoning to the position and orientation of the objects, give more reasoning to the directional description for the robot dog and target object.\n\n"
                    "**Robot Dog Identification:**\n"
                    "- The robot dog has a black area near its head and a red area near its tail, which can help you determine its position and orientation.\n"
                    "- The position of the robot dog is represented by the coordinates of its head (black area).\n"
                    "- When the robot dog's head is oriented toward the positive y-axis (Front/Forward orient), its orientation is 0 degrees.\n\n"
                    "**Orientation Measurement:**\n"
                    "- Orientation is measured in degrees from -180 to 180.\n"
                    "- 0 degrees points along the positive y-axis (forward).\n"
                    "- **Positive angles** indicate a **clockwise** rotation from the positive y-axis.\n"
                    "- **Negative angles** indicate a **counterclockwise** rotation from the positive y-axis.\n\n"
                    "**Examples:** Task: Move the robot to the green cube.\n"
                    "    - robot dog: position: (1.0, 3.0), orientation: 0\n"
                    "    - brown cube: position: (-3.5, 2.5), orientation: 0\n"
                    "    - green cube: position: (-1.5, 1.5), orientation: 0\n"
                )
            },
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": f" Please describe the environment according to the assigned robot."},
                    {"type": "image_url", 
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}",
                        },
                    }
                ],
            },
        ]
    return value

prompt_current_target_orientation = [
    {
        "role": "system",
        "content": (
            "You are tasked with choosing the correct orientation for the robot dog to face the target object. The robot dog's orientation is represented in degrees, with 0° indicating the robot dog is facing forward along the positive y-axis.\n\n"
            "The positive x-axis points to the right, and the positive y-axis points forward. The orientation is measured in degrees from -180° to 180°, with positive angles indicating a clockwise rotation from the positive y-axis.\n\n"
            "  **Relative Position Calculation Method**:\n\n"
            "    a. **Calculate Coordinate Differences**:\n\n"
            "     - `dx = target object's x-coordinate - robot's x-coordinate`\n"
            "     - `dy = target object's y-coordinate - robot's y-coordinate`\n\n"
            "    b. **Calculate Orientation Angle** (in degrees):\n\n"
            "     - `angle = atan2(dx, dy) * (180 / π) (dy > 0: |angle| < 90; dy < 0: |angle| > 90 )`\n"
            "     - Note: The `atan2` function returns an tan angle of dx/dy, the value is between -180° and 180°.\n\n"
            "    c. **Determine Relative Direction**:\n\n"
            "     - **target_orientation = 0**: `-22.5° < angle ≤ 22.5°`\n"
            "     - **target_orientation = 45**: `22.5° < angle ≤ 67.5°`\n"
            "     - **target_orientation = 90**: `67.5° < angle ≤ 112.5°`\n"
            "     - **target_orientation = 135**: `112.5° < angle ≤ 157.5°`\n"
            "     - **target_orientation = 180**: `angle > 157.5°` or `angle ≤ -157.5°`\n"
            "     - **target_orientation = -135**: `-157.5° < angle ≤ -112.5°`\n"
            "     - **target_orientation = -90**: `-112.5° < angle ≤ -67.5°`\n"
            "     - **target_orientation = -45**: `-67.5° < angle ≤ -22.5°`\n\n"
            "Your output should only keep in the following json format\n\n"
            "{\n"
            '  "current orientation": 0,\n'
            '  "target orientation": 45\n'
            "}"
        )
    },
]

prompt_current_target_orientation_detach = [
    {
        "role": "system",
        "content": (
            "You are tasked with making the object's orientation to 0.\n\n"
            'Example: The object\'s orientation is 45. So the current orientation is 45 and the target orientation is 0\n'
            "Your output should only keep in the following json format\n\n"
            "{\n"
            '  "current orientation": 45,\n'
            '  "target orientation": 0\n'
            "}"
        )
    },
]



## Low-level
prompt_drone = [
    {
        "role": "system",
        "content": (
            "You are responsible for choose the motion function to finish the task.\n"
            "Do not generate any prefix or suffix for the motion function.\n\n"
            "Motion Functions:\n"
            "1. `self.drone_motions.construct_map()`\n"
            "   - Controls the drone to construct a global map.\n"
        ),
    },
]

prompt_robot_dog = [
    {
        "role": "system",
        "content": (
            "You are responsible for choose the motion function to finish the task.\n"
            "Do not generate any prefix or suffix for the motion function.\n\n"
            "**Motion Function Library**:\n\n"
            "1. `self.dog_motions.go1_attach_start('name_of_target_object')`\n"
            "   - name of target object. Name of the object need to be attached. Example: 'A_letter_cube', 'green_cube'\n"
            "2. `self.dog_motions.go1_detach_start('name_of_target_object')`\n"
            "   - name of target object. Name of the object need to be detached. Example: 'B_letter_cube', 'blue_cube'\n"
        )
    }
]

prompt_robots_cooperation = [
    {
        "role": "system",
        "content": (
            "You are responsible for choose the motion function to finish the task.\n"
            "For some tasks, it can be finish by the cooperation of robots.\n\n"
            "Do not generate any prefix or suffix for the motion function.\n\n"
        ),
    },
    {
        "role": "system",
        "content": (
            "Motion Functions comb:\n"
            "     - Fixed Usage Example:\n"
            "         drone_thread = threading.Thread(target=self.drone_motions.quad_planning_start, args=('task contain',))\n"
            "         dog_thread = threading.Thread(target=self.dog_motions.go1_following_start, args=('task contain',)\n"
            "         drone_thread.start()\n"
            "         dog_thread.start()\n"
            "         drone_thread.join()\n"
            "         dog_thread.join()\n\n"
            "   - Parameters:\n"
            "     - task contain: The task need to be finish. Drone and dog share same contain\n"
            "   - Coding Guidelines:\n"
            "     - You must create multiple threads to finish the task, because each robot need to work in their own thread\n"  
        ),
    },
]

from openai import OpenAI

def prompt_assistant(task):
    client = OpenAI()
    name = client.chat.completions.create(
        model='gpt-4o-mini',
        messages = [ 
                    {
                        "role": "system",
                        "content": (
                            "You are responsible for decide which prompt need to be used in the task.\n\n"
                            "If the task is related to the drone, output the prompt for the drone. (prompt_drone)\n"
                            "If the task is related to the robot dog, output the prompt for the robot dog. (prompt_robot_dog)\n"
                            "If the task is related to the cooperation of robot dog and drone, output the prompt for the cooperation. (prompt_robots_cooperation)\n"
                            "Your output should only include the prompt name]"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"The task is {task}, which prompt should be used?"
                        ),
                    },
                ],
        temperature=0.1,
    )
    return name.choices[0].message.content

def prompt_component(img_base64, img_base64_norm): 
    value= [
            {
                "role": "system",
                "content": (
                    "You are an environment describer. You will be given a top-down view of an environment with a grid ranging from (-5.0, -3.0) to (5.0, 3.0), where (-5.0, -3.0) is the rear-left corner and (5.0, 3.0) is the front-right corner. Please carefully examine the image and provide the coordinates for each object in the environment.\n\n"
                    "Only output the object you can see and you can recognize with high confidence. If you can not recognize the object prefectly, you can ignore it.\n\n"
                    "You will see the shadow of the UAV in the image, you can ignore it.\n\n"
                    "**Robot Dog Identification:**\n"
                    "- The robot dog has a black area near its head and a red area near its tail, which can help you determine its position and orientation.\n"
                    "- The position of the robot dog is represented by the coordinates of its head (black area).\n"
                    "- When the robot dog's head is oriented toward the positive y-axis (Front/Forward orient), its orientation is 0 degrees.\n\n"
                    "- Your output should follow the example:\n\n"
                    "    - brown cube: position: (-3.5, 2.5)\n"
                    "    - green cube: position: (-1.5, 1.5)\n"
                )
            },
            {
                "role": "user",
                "content": (
                    "I provided you two images, one is the original image, the other is the grid image. The original image is used to help you identify the object. The grid image is the original image with grid, which will be helpful for coordinates and orientation detection. Please describe the environment according to the original image."
                ),
            },
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": " This is original image."},
                    {"type": "image_url", 
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64_norm}",
                        },
                    }
                ],
            },
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": " This is grid image."},
                    {"type": "image_url", 
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}",
                        },
                    }
                ],
            },
        ]
    return value

prompt_constructor = [
    {
        "role": "user",
        "content": (
            "You are tasked with constructing a global world coordinate system based on local coordinates derived from multiple images.\n\n"
            "Inputs you will receive:\n"
            "1. **Local Coordinates**: The positions of various objects within each individual image.\n"
            "2. **Image Center Coordinates**: The world coordinates corresponding to the center point of each image.\n\n"
            "Your responsibilities include:\n"
            "1. **Integrate Coordinates**: Combine the local coordinates of objects with the world coordinates and yaw angle of the image centers to determine each object's position in the global world coordinate system.\n"
            "2. **Identify Unique Objects**: When objects share the same name across different images, they are same entity, but the coordinates may be different. You need to integrate these coordinates to find the best coordinate to reprensent it.\
            If the difference of two coordinates of one object are too big, you need to choose the correct one, the correct one always the one close to the central of its own local map, because when the object close to the boundary, the detection result might be wrong.\n\n"
            "Guidelines:\n"
            "- Ensure accurate alignment of local coordinates to the global system.\n"
            "- Present the final world coordinates in a clear and organized manner.\n"
            "- You do not need to care about the orientation of the image, it also is 0\n"
            "- The final output should include the global coordinates of each object in the world coordinate system.\n"
            "- Sometime the same object will descirbe in different name but same meaning, you need to integrate them to one object. For example, cube and block share same meaning. So, E cube and E block in differnet local map actually is one thing.\n"
            "- You must overlook the red wire and red hand and tool box while constructing the global coordinate system.\n\n"
        ),
    },
]

# def prompt_local_directon(img_base64, img_base64_norm, target_mission): 
#     value= [
#             {
#                 "role": "system",
#                 "content": (
#                     "You are an distance describer. You will be given a top-down view of an environment with a grid ranging from (-5.0, -3.0) to (5.0, 3.0), where (-5.0, -3.0) is the rear-left corner and (5.0, 3.0) is the front-right corner.\n\n"
#                     "**Robot Dog Identification:**\n"
#                     "- The robot dog has a black area near its head and a red area near its tail, which can help you determine its position and orientation.\n"
#                     "- The position of the robot dog is represented by the coordinates of its head (black area).\n\n"
#                     "- You can see the shadow of the UAV in the image, you can ignore it.\n\n"
#                     "Please provide the the main object coordinate and the obstacles coordinate in Json format and never overlook obstacles.\n\n"
#                     "Example 1: The task is 'Move to the A letter cube'. The main object in this task is robot. robot position: (2.0, -1.0), A letter position: (0.0,0.0). A cube is the destination, so it is not the obstacle ad there are no other objects, so the obstacles position can be null. \
#                      Your output should inlcude the following information\n\n"
#                     "{\n"
#                     '  "object": "robot",\n'
#                     '  "object position": [2.0, -1.0],\n'
#                     '  "obstacles position": null,\n'
#                     '  "target position": null\n'
#                     "}"
#                     "\nExample 2: The task is 'Carry the A cube to the left side of B cube'. Main object is A cube, robot position: (2.0, -1.0), A cube position: (2.0, 0.0), B cube position: (4.0, 3.0), C cube position: (3.0, 5.0). \
#                      Because target point is (4.0-2.0, 3.0), B letter position and C letter can be obstacle. A letter is the carried thing that will move together with the robot, so in the carring task robot is not obstacle.\n\n\
#                      Your output should inlcude the following information\n\n"
#                     "{\n"
#                     '  "object": "A cube",\n'
#                     '  "object position": [2.0, 0.0],\n'
#                     '  "obstacles position": [[4.0, 3.0], [3.0, 5.0]],\n'
#                     '  "target position": null\n'
#                     "}"
#                     "\nExample 3: The task is 'Move to the A letter cube'. If the you already can see the target position in the local image, you can add the target position.\n\
#                      Your output should inlcude the following information\n\n"
#                     "{\n"
#                     '  "object": "robot",\n'
#                     '  "object position": [2.0, -1.0],\n'
#                     '  "obstacles position": [[4.0, 3.0], [3.0, 5.0]],\n'
#                     '  "target position": [0.0, 0.0],\n'
#                     "}"
#                     "\nNotice 1: Local image can not identify the relative direction in world coordinate, because it has its own rotation. So you can not decide the target postion even you can see the target if the task include relative position.\n\n"
#                     "\nNotice 2: The input coordinate in task is the world coordinate, but the target posiition you need to output is the local coordinate, so never treate the world coordinate from the task as the target position. (Just leave the target position as null)\n\n"
#                 )
#             },
#             {
#                 "role": "user", 
#                 "content": [
#                     {"type": "text", "text": f"Current task is {target_mission}.\nI provided you two images, one is the original image, the other is the grid image. The original image is used to help you identify the object. The grid image is the original image with grid, which will be helpful for coordinates and orientation detection. Please describe the environment according to the original image."},
#                 ],
#             },
#             {
#                 "role": "user", 
#                 "content": [
#                     {"type": "text", "text": " This is original image."},
#                     {"type": "image_url", 
#                     "image_url": {
#                         "url": f"data:image/jpeg;base64,{img_base64_norm}",
#                         },
#                     }
#                 ],
#             },
#             {
#                 "role": "user", 
#                 "content": [
#                     {"type": "text", "text": " This is grid image."},
#                     {"type": "image_url", 
#                     "image_url": {
#                         "url": f"data:image/jpeg;base64,{img_base64}",
#                         },
#                     }
#                 ],
#             },
#         ]
#     return value

def prompt_local_directon(img_base64, img_base64_norm, target_mission): 
    value= [
            {
                "role": "system",
                "content": (
                    "You are an distance describer. You will be given a top-down view of an environment with a grid ranging from (-5.5, -3.5) to (5.5, 3.5), where (-5.5, -3.5) is the rear-left corner and (5.5, 3.5) is the front-right corner.\n\n"
                    "**Robot Dog Identification:**\n"
                    "- The robot dog has a black area near its head and a red area near its tail, which can help you determine its position and orientation.\n"
                    "- The position of the robot dog is represented by the coordinates of its head (black area).\n"
                    "- You can see the shadow of the UAV in the image, you can ignore it.\n"
                    "- Do not treat the target as obstacle.\n\n"
                    "Please provide the the main object coordinate and the obstacles coordinate in Json format and never overlook obstacles.\n\n"
                    "Example 1: The task is 'Move to the A letter cube'. The main object in this task is robot. If you can not see the A cube, robot position: (2.0, -1.0), A letter position: (5.0,5.0). A cube is the destination and may not to be catched, so it is not the obstacle ad there are no other objects, so the obstacles position can be null. \
                     Your output should inlcude the following information\n\n"
                    "{\n"
                    '  "object": "robot",\n'
                    '  "object position": [2.0, -1.0],\n'
                    '  "obstacles position": null,\n'
                    '  "target position": null\n'
                    "}"
                    "\nExample 2: The task is 'Carry the A cube to the left side of B cube'. Main object is A cube, robot position: (2.0, -1.0), A cube position: (2.0, 0.0), B cube position: (4.5, 3.0), C cube position: (3.0, 5.0). \
                     Because target point is (4.5-2.0, 3.0) (at least keep 2 units), B letter position (Though it is used for target navigation, but it still can be obstacle) and C letter can be obstacle. A letter is the carried thing that will move together with the robot, so in the carring task robot is not obstacle.\n\n\
                     Your output should inlcude the following information\n\n"
                    "{\n"
                    '  "object": "A cube",\n'
                    '  "object position": [2.0, 0.0],\n'
                    '  "obstacles position": [[4.5, 3.0], [3.0, 5.0]],\n'
                    '  "target position": [2.5, 3.0]\n'
                    "}"
                )
            },
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": " This is the original image."},
                    {"type": "image_url", 
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64_norm}",
                        },
                    }
                ],
            },
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": " This is the grid image."},
                    {"type": "image_url", 
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}",
                        },
                    }
                ],
            },
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": f"Current task is {target_mission}.\nI provided you two images, one is the original image, the other is the grid image. Please describe the environment according to the two images."},
                ],
            },
        ]
    return value

prompt_global_classification = [
    {
        "role": "system",
        "content": (
            "You are tasked with classifying the objects in the environment. According to the task to make decision.\n\n"
            "   - target_position: The current position of the target. Example: [0,0]\n"
            "   - obstacle positions: The current position of the other objects which can be obstacle. (Except the robot dog) Example: [[0,0],[1,1]]\n"
            "According to the task, adjust the output.\n"
            "Example:\n\n"
            "Task: Carry the A to left side of B [0,2]. Target position will be [-2, 2] (Substract 2 units in x-axis 0-2 = -2), obstacle positions are [[0,2] (B need to be treated as an obstacle, because the path need avoid B to get the target position) (A is not treated as obstacle, because it is carried thing)]\n"
            "You output should only include the following information and keep the json format. Notice: Do not use (), use []\n\n"
            "{\n"
            '  "target position": [3, 4],\n'
            '  "obstacle positions": [[5, 6], [7, 8]]\n'
            "}"
        ),
    },
]