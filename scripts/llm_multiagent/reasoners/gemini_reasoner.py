import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting
import time
import threading
import yaml
from pathlib import Path
import json

class GeminiReasoner:
    def __init__(self):
        cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        vertexai.init(project=cfg["project_id"],
                        location="us-central1")
        self.constructor_model = GenerativeModel(
            cfg["pro_model"],
            system_instruction="""
            Construct a global map by logically integrating all provided local map data. Each object appears exactly once globally. Each local map has a detectable area ranging from (-7, -4) to (7, 4), corresponding to the camera’s field of view. However, the global map is constructed by logically integrating all local maps, not limited to a single local map’s boundaries.
            Answer concisely and rapidly, without unnecessary overthinking.
            **Prioritize Logical Consistency:**
            1. **Cross-map Validation:**
            - Always cross-check objects across multiple local maps; do not rely on a single observation. The camera's field of view can support the observation range from (-7, -4) rear-left corner to (7, 4) front-right corner, That means the objects in the range (-7 + x, -4 + y) to (7 + x, 4 + y) could be observed in the local map, where (x, y) is the position of the camera when capturing the local map.
            2. **Error Identification:**
            - Detect and exclude clearly erroneous entries (e.g., objects appearing only once in improbable positions).
            - If an object is only present in a single local map and absent from others where logically it should be visible, remove it.
            3. **Conflict Resolution:**
            - Resolve conflicting labels (e.g., 'L' vs. 'E') by majority voting across maps or flag as uncertain if unresolved.
            - Same objects could have error in position across different maps. However, I believe the error is below 5 units. If there are huge gap, you need to check other maps to find the correct position.
            4. **Synonym Handling:**
            - Recognize synonymous labels (e.g., 'Block_L' = 'L_Cube').
            5. **Clarify 'Unknown' Labels:**
            - Do not include 'Unknown' labels; resolve using clearer identifications from other maps.

            **Explicit Step-by-Step Reasoning:**
            - Step 1: Verify object positions consistency across all maps.
            - Step 2: Resolve all label conflicts clearly.
            - Step 3: Perform a final proximity check (less than 2 units apart) and correct issues logically.

            **Output Requirement:**
            Present your thought chain and clearly state results in the format below: (Only use "{" and "}" to enclose the final results)
            {    
                Name: XXX, Coordinate: (X, Y),
                Name: YYY, Coordinate: (X, Y),
                ...
            }   
            """
        )

        self.updator_model = GenerativeModel(
            cfg["pro_model"],
            system_instruction="""
            Update the existing global map using the latest local map data. Update object names and coordinates explicitly without averaging.

            **Instructions:**
            1. **Identify Local Errors:**
            - Local map entries may contain misidentifications; carefully validate before updating.
            2. **No New Additions:**
            - Do not introduce new objects; treat any new local detections as potential errors.
            3. **Letter Shape Priority:**
            - Prioritize matching objects by visual letter similarity before positional accuracy (e.g., L/T/I, V/A/U).
            4. **Resolving Conflicts:**
                - Prefer existing global map entries in conflicts unless strong visual evidence justifies updating.
                - If multiple possible matches arise, select the match causing minimal structural change, prioritizing letter shape consistency.

            **Output Format:**
            Clearly state only the updated global map:
            Name: XXX, Coordinate: (X, Y)
            """
        )

        self.global_planner_model = GenerativeModel(
            cfg["pro_model"],
            system_instruction="""
            You are a task decomposer for a heterogeneous multi-robot system. Decompose complex tasks clearly into sub-tasks suitable for each robot's specific abilities.

            **Workspace Definition:**
            - Range: (-11, -6) rear-left corner to (11, 6) front-right corner.
            - When you assemble word, consider how much the word occupies in the workspace. The distance between each cube is 4 units. The center cubes of the word should close to the center of the workspace.
              For example, when assembling the word 'ROCK' from left to right, the center of the word should be at (0, 0). So the R cube should be at (-6, 0), O cube should be at (-2, 0), C cube should be at (2, 0), and K cube should be at (6, 0). 
              But when you assemble from right to left, the initial cube should be at (-6, 0), to make the center of the word at (0, 0). 
              Similarly, when assembling from front to back, the initial cube should be at (0, 6). When assembling from back to front, the initial cube should be at (0, -6).
            **Positioning Rules:**
            - Clearly specify initial placement coordinates for cubes when arranging words. But the rest cubes could be placed use the related position to previous cubes.
            - Default arrangement direction is from left to right, but you can also arrange from right to left or top to bottom if explicitly requested or logically beneficial.
            - Ensure the entire word is centrally positioned within the workspace.
            - If specific cubes are requested to remain stationary, clearly state the adjusted arrangement sequence accordingly.
            - Assume you already know the position of the cubes, so if you want to move to the cube, never use the exact coordinates, but directly use the cube name.

            **Robot Abilities:**
            - Drone:
                1. Construct the map (Always first).
            - Robot Dog:
                1. Attach to object (after reaching position).
                2. Detach from object (after transportation).
            - Drone and Robot Dog Cooperation:
                - Move to target (Drone plans path, Robot Dog follows).
                - Carry object to target (Drone plans path, Robot Dog carries).

            **Task Decomposition Rules:**
            - Movement tasks always involve Drone-Dog cooperation.
            - Do not modify simple tasks ('Move to...', 'Carry...').
            - Explicitly define cube placement sequences when assembling words.

            **Output Format:**
            Strictly output decomposed tasks as:
            [[Robot Name]: 'Task1'; [Robot Name]: 'Task2'; ...]

            **Examples:**
            Task: Carry Object A to Location B
            [[Drone]: 'Construct the map.'; [Drone and Robot Dog]: 'Move to Object A location.'; [Robot Dog]: 'Attach to Object A.'; [Drone and Robot Dog]: 'Carry Object A to Location B.'; [Robot Dog]: 'Detach from Object A.']

            Task: Assemble 'ROCK', keeping 'K' stationary
            Thought:  Skip the K cube, move 'C' left of 'K', then 'O' left of 'C', and finally 'R' left of 'O'.

            **Error Handling:**
            If unclear, explicitly state:
            "Unable to decompose the task."
            """
        )

        self.mf_planner_model = GenerativeModel(
            cfg["flash_model"],
            system_instruction= "You are responsible for choose the motion function to finish the task.\n"
                                "You only need to output the contain I wroten\n\n"
                                "Do not output ```python or ``` or any other prefix or suffix\n\n"
                                "Drone Motion Functions:\n"
                                "1. `self.drone_motions.construct_map()`\n"
                                "   - Controls the drone to construct a global map.\n"
                                "Robot Dog Motion Functions:\n\n"
                                "1. `self.dog_motions.go1_attach_start('name_of_target_object')`\n"
                                "   - name of target object. Name of the object need to be attached. Example: 'A_letter_cube', 'green_cube'\n"
                                "2. `self.dog_motions.go1_detach_start('name_of_target_object')`\n"
                                "   - name of target object. Name of the object need to be detached. Example: 'B_letter_cube', 'blue_cube'\n"
                                "Cooperation Motion Functions:\n"
                                "   - Fixed Usage:\n"
                                "        'drone_thread = threading.Thread(target=self.drone_motions.quad_planning_start, args=('task contain',))\n"
                                "         dog_thread = threading.Thread(target=self.dog_motions.go1_following_start, args=('task contain',))\n"
                                "         drone_thread.start()\n"
                                "         dog_thread.start()\n"
                                "         drone_thread.join()\n"
                                "         dog_thread.join()'\n\n"
                                "   - Parameters:\n"
                                "     - task contain: The task need to be finish. Drone and dog share same contain\n"
        )

        self.observation_range_model = GenerativeModel(
            cfg["flash_model"],
            system_instruction="""
            Given image positions, calculate global observation ranges based on local detection range (-12,-6.5) to (12,6.5).

            Output clearly:
            Image X: Observation Range: from global (X1,Y1) to (X2,Y2)
            """
        )

        self.global_coordinate_model = GenerativeModel(
            cfg["flash_model"],
            system_instruction="""
            Given each image's position and objects' local coordinates, calculate each object's global coordinates.

            Output clearly per image:
            Image X:
                - Object Name: Global Coordinate: (X,Y)
            """
        )

        self.table_integration_model = GenerativeModel(
            cfg["flash_model"],
            system_instruction="""
            Given observation ranges and global object coordinates from each image, integrate them into a table:
            - Rows: Image number and its observation range.
            - Columns: Objects detected across images with global coordinates.

            If you meet the different object names but the same object, please use the same name to represent the object. For example, 'Block_L' and 'L_Cube' are the same object, you can use 'Block_L' to represent the object.
            You can overlook the object named with 'Unknown'.

            Output in a clear table format. If the object is not detected in an image, leave the cell with (none, none). Only use one space to separate the coordinate.

            Example:
            | Image | Visible Range | 'obj name' | 'obj name' | 'obj name' | 'obj name' | 'obj name' |
            |---|---|---|---|---|---|---|---|---|
            | Image 1 (XX, XX) | (XXX) ~ (XXX) | (XXX,XXX) | (XXX, XXX) | (none, none) | (none, none) | (none, none) |
            """
        )

        self.table_renewal_model = GenerativeModel(
            cfg["flash_model"],
            system_instruction="""
            Update the table with the error checking results.
            There are three checkers:
            - Inconsistencies Checker: It checks the coordinate inconsistencies between images. Normally, you should remove the isolated data if there are one data conflict with others. If only two data conflict with each other, you can check other checkers, find if one of them is mislabeled or misdetected. If one is mislablled or misdetected, the another one is correct, so you can keep the correct one.
            - Mislabeled Object Checker: It checks the mislabeled objects.
            - Misdetected Object Checker: It checks the misdetected objects. Remove the misdetected object suggested by the checker.

            Also, you need to unify the non-unique object names. For example, 'Block_L' and 'L_Cube' are the same object, you can use 'Block_L' to represent the object.

            Analyze the three checkers's resluts together and update the table.

            Output format:
            | Image | Visible Range | 'obj name' | 'obj name' | 'obj name' | 'obj name' | 'obj name' |
            |---|---|---|---|---|---|---|---|---|
            | Image 1 (XX, XX) | (XXX) ~ (XXX) | (XXX,XXX) | (XXX, XXX) | (none, none) | (none, none) | (none, none) |
            """
        )

        self.inconsistencies_checker_model = GenerativeModel(
            cfg["flash_model"],
            system_instruction="""
            Analyze the provided table thoroughly to identify coordinate inconsistencies. You can overlook the (none, none) coordinates.
            
            **Instructions:**
            - Compare each object's coordinates across all images.
            - If the coordinates of the same object vary significantly (more than 3 units apart) between images, clearly identify which image's data is likely incorrect based on majority consistency.

            **Notice:**
            - Do not set the (none, none) coordinates the standard for inconsistency, as they are not detected objects.
            - If the object only have two conflicting coordinates, you should mention the object have confilicting coordinates in the output.

            **Output contain:**
            Output the abnormal situation in the table.  For example:
            'object name': In image XX the coordinate is (X1, Y1), in image YY and ZZ the coordinates are around (X2, Y2), the distance is sqrt((X1-X2)^2 + (Y1-Y2)^2) > 3. So this object's coordinates are inconsistent.
            ...
            Summary:
            The object 'object name' has inconsistent coordinates. You should check the data in image XX, YY. You should delet the data in image XX or the data in images YY and ZZ. But at least keep one of them.

            If there are no inconsistencies, output 'No inconsistencies found.'
            """
        )

        self.mislabeled_object_checker_model = GenerativeModel(
            cfg["flash_model"],
            system_instruction="""
            Analyze the provided table carefully to identify and correct mislabeled or Unknown objects. Sometimes, the labeler will label the object to similar object. For example, 'L' is similar to 'E', so the labeler may label the 'L' to 'E' by mistake.

            **Instructions:**
            - Find the similar coordinate all of the images and objects, and if they are not in the same column, you need to correct the table.
            
            **Notice:**:
            - Do not compare the coordinates of the same object in different images.
            - The main check target is the object with similar appearance, for example, 'L', 'B' and 'E' are similar, 'A' and 'V' are similar, 'L' and 'I' and 'T' are similar.

            **Output Format:**
            Analysis:
                object name
                   Image 1: (1.0, 7.0)
                   Image 2: (-3.5, -1.8)
                   The coordinates of 'object name' in Image XXX and Image YYY are significantly different.
                   In Image ZZZ, the coordinate of 'object name2' is (-3.9, -1.6), which is similar to the coordinate of 'object name' in Image 2.
                   So, the 'object name' in Image 2 is likely to be 'object name2'.   
            **Summary**:
                Based on the coordinate analysis and shape similarities, the following object is likely to be mislabeled:
                - 'object name' in Image 2 should be corrected to 'object name2' and the 'object name' in Image 1 should be reasonable. (Must tell me the correct object name)
            """
        )

        self.misdetected_object_checker_model = GenerativeModel(
            cfg["flash_model"],
            system_instruction="""
            Find the object have equal or more than 4 (none, none) coordinates in the table.
            If no objects are found, clearly output:
                **No misdetected objects found.**
            If you find the object, output the object name and the image number where the object is found.
            If you can not find, output None.
            Output format:
                'object name'
            """
        )

        self.misdetected_visible_checker_model = GenerativeModel(
            cfg["flash_model"],
            system_instruction="""
            Find the provided object if visible in all images. (except the image can found the object)
            If the input is None, output 'No misdetected objects found.'
            Output format:
                Name: [Object Name], Coordinate: (X,Y)
                - Image [Number] Visible Range: (X1,Y1) to (X2,Y2), Should be visible: Yes/No
                ... (repeat for each image except the detection image)
            """
        )

        self.misdetected_object_assistant_model = GenerativeModel(
            cfg["flash_model"],
            system_instruction="""
            Check the 'Should be visible' value, if all of the 'Should be visible' value is 'No', you can keep the object from the table. Otherwise, remove the object in the table.
            
            Sometimes, the input will not include the 'Should be visible' value, but said 'No misdetected objects found.' That means all of the objects should be kept in the table.

            Output clearly:
            **Name: XXX, Decision: Remove**

            If no object should be removed, output **No object should be removed.**
            """
        )

        self.global_map_analysis_model = GenerativeModel(
            cfg["flash_model"],
            system_instruction="""
            According to the table and output the final global map.
            Each object only have one in the environment.
            After utilizing the table and error checking, output the final global map. You can use the average and correct data to update the global map.
            If the object in the table, but all coordinates are (none, none), you can exclude the object from the global map.

            Output clearly:
            Name: XXX, Coordinate: (X,Y)
            """
        )

        self.generation_config = {
                    "max_output_tokens": 16384,
                    "temperature": 0.0,
                    "top_p": 0.1,
                }
        self.generation_config_json = {
                    "max_output_tokens": 16384,
                    "temperature": 0.1,
                    "top_p": 0.5,
                    "response_mime_type": "application/json",
                    "response_schema": {"type":"OBJECT","properties":{"reason": {"type":"STRING"}, "answer": {"type":"STRING"}}, "required":["reason"]},
                }
        self.safety_settings = [
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
        ]
        self.global_map = ''        

    def global_planner(self, message_t):
        contents = [
            {
                "role": "user",
                "parts": [
                    {
                        "text": "If the destination is the location of the cube, you don't need to give the exact coordinates. Just give the relative position of the cube. For example, 'move to the location of X cube' or 'move to the right side of the cube'."
                    },
                    {
                        "text": "Do not add any prefix or suffix."
                    },
                    {
                        "text": f"Please decompose the task {message_t}. The format of your output should only include the contains shown in the following format: [[robot_name]: 'Task1'; [robot_name]: 'Task2'; [robot_name]: 'Task3'; ...]."
                    }
                ]
            }
        ]
        try:
            while True:
                response = self.global_planner_model.generate_content(
                    contents=contents,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings
                )
                decision_text = response.text
                break
        except KeyboardInterrupt:
            print("Keyboard interrupt received, exiting...")
            raise
        except Exception as e:
            print("Meet error but I will try again after 2s. Reason:", e)
            print(f"Retry after 2s.")
            time.sleep(2)
        return decision_text
    
    def mf_planner(self, command):
        contents = [
            {
                "role": "user",
                "parts": [
                    {
                        "text": f"The task is {command}.\nPlease generate the motion function(s) to finish the sub-task. Do not use quotes or any other prefix or suffix."
                    }
                ]
            }
        ]
        try:
            while True:
                response = self.mf_planner_model.generate_content(
                    contents=contents,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings
                )
                decision_text = response.text
                decision_text = decision_text.replace('```', '')
                decision_text = decision_text.replace('python', '')
                break
        except KeyboardInterrupt:
            print("Keyboard interrupt received, exiting...")
            raise
        except Exception as e:
            print("Meet error but I will try again after 2s. Reason:", e)
            print(f"Retry after 2s.")
            time.sleep(2)
        return decision_text
    
    def map_constructor(self, text):
        contents = [
            {
                "role": "user",
                "parts": [
                    {
                        "text": text
                    },
                    {
                        "text": "You need to make your answer compact and clear. Your output will be added into the a comprehensive analysis with other answers and then sent to the final integrator model."
                    }
                ]
            }
        ]
        while True:
            try:
                response = self.table_integration_model.generate_content(
                    contents=contents,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings
                )
                table = response.text
                break
            except Exception as e:
                print("Meet error but I will try again after 2s. Reason:", e)
                print(f"Retry after 2s.")
                time.sleep(2)
        contents = [
            {
                "role": "user",
                "parts": [
                    {
                        "text": f'{text}Please construct the global map based on all available local map data.'
                    }
                ]
            }
        ]
        try:
            while True:
                response = self.constructor_model.generate_content(
                    contents=contents,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings
                )
                self.global_map = response.text
                # Extract content within {} brackets
                start_idx = response.text.find('{')
                end_idx = response.text.rfind('}')
                if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                    self.global_map = response.text[start_idx+1:end_idx].strip()
                else:
                    self.global_map = response.text
                break
        except KeyboardInterrupt:
            print("Keyboard interrupt received, exiting...")
            raise
        except Exception as e:
            print("Meet error but I will try again after 2s. Reason:", e)
            print(f"Retry after 2s.")
            time.sleep(2)
        return response.text
    
    def map_constructor_old(self, text):
        self.text2table = ""
        self.error_checker = ""
        self.final_table = ""
        self.misdetected_explain = ""
        def summary_extractor(text):
            lines = text.split('\n')
            summary_lines = []
            in_summary = False
            
            for line in lines:
                if "Summary:" in line and not in_summary:
                    in_summary = True
                elif in_summary:
                    # Check if we've reached the end of the summary section
                    if not line.strip() == "":
                        summary_lines.append(line)
            
            return "\n".join(summary_lines) if summary_lines else ""
        def observation_range(text):
            contents = [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": text
                        },
                        {
                            "text": "You need to make your answer compact and clear. Your output will be added into the a comprehensive analysis with other answers and then sent to the final integrator model."
                        }
                    ]
                }
            ]
            try:
                while True:
                    response = self.observation_range_model.generate_content(
                        contents=contents,
                        generation_config=self.generation_config,
                        safety_settings=self.safety_settings
                    )
                    break
            except KeyboardInterrupt:
                print("Keyboard interrupt received, exiting...")
                raise
            except Exception as e:
                print("Meet error but I will try again after 2s. Reason:", e)
                print(f"Retry after 2s.")
                time.sleep(2)
            self.text2table += (response.text + '\n\n')        
        def global_coordinate(text):
            contents = [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": text
                        },
                        {
                            "text": "You need to make your answer compact and clear. Your output will be added into the a comprehensive analysis with other answers and then sent to the final integrator model. Do not add any prefix or suffix."
                        }
                    ]
                }
            ]
            try:
                while True:
                    response = self.global_coordinate_model.generate_content(
                        contents=contents,
                        generation_config=self.generation_config,
                        safety_settings=self.safety_settings
                    )
                    break
            except KeyboardInterrupt:
                print("Keyboard interrupt received, exiting...")
                raise
            except Exception as e:
                print("Meet error but I will try again after 2s. Reason:", e)
                print(f"Retry after 2s.")
                time.sleep(2)
            self.text2table += (response.text + '\n\n')
        def inconsistencies_checker(text):
            contents = [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": text
                        },
                        {
                            "text": "Give a completed analysis of each object in the table. "
                        }
                    ]
                }
            ]
            try:
                while True:
                    response = self.inconsistencies_checker_model.generate_content(
                        contents=contents,
                        generation_config=self.generation_config,
                        safety_settings=self.safety_settings
                    )
                    break
            except KeyboardInterrupt:
                print("Keyboard interrupt received, exiting...")
                raise
            except Exception as e:
                print("Meet error but I will try again after 2s. Reason:", e)
                print(f"Retry after 2s.")
                time.sleep(2)
            self.error_checker += ("Inconsistencies Check\n" + summary_extractor(response.text) + '\n\n')
        def mislabeled_object_checker(text):
            contents = [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": text + '\n'
                        },
                        {
                            "text": "Give a completed analysis of each object in the table. "
                        }
                    ]
                }
            ]
            try:
                while True:
                    response = self.mislabeled_object_checker_model.generate_content(
                        contents=contents,
                        generation_config=self.generation_config,
                        safety_settings=self.safety_settings
                    )
                    break
            except KeyboardInterrupt:
                print("Keyboard interrupt received, exiting...")
                raise
            except Exception as e:
                print("Meet error but I will try again after 2s. Reason:", e)
                print(f"Retry after 2s.")
                time.sleep(2)
            self.error_checker += ("Mislabeled Check\n" + summary_extractor(response.text) + '\n\n')
        def misdetected_object_checker(text):
            contents = [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": text
                        },
                        {
                            "text": "Give a completed analysis of each object in the table. "
                        }
                    ]
                }
            ]
            try:
                while True:
                    response = self.misdetected_object_checker_model.generate_content(
                        contents=contents,
                        generation_config=self.generation_config,
                        safety_settings=self.safety_settings
                    )
                    break
            except KeyboardInterrupt:
                print("Keyboard interrupt received, exiting...")
                raise
            except Exception as e:
                print("Meet error but I will try again after 2s. Reason:", e)
                print(f"Retry after 2s.")
                time.sleep(2)
            # print(response.text)
            explain = explain_misdetected_object(response.text)
            self.error_checker += ("Misdetected Check\n" + explain + '\n\n')
        def renew_table(table):
            contents = [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": "This is the original table:\n" + table
                        },
                        {
                            "text": "This is the decisions from three checkers:\n" + self.error_checker
                        }
                    ]
                }
            ]
            try:
                while True:
                    response = self.table_renewal_model.generate_content(
                        contents=contents,
                        generation_config=self.generation_config,
                        safety_settings=self.safety_settings
                    )
                    table = response.text
                    break
            except KeyboardInterrupt:
                print("Keyboard interrupt received, exiting...")
                raise
            except Exception as e:
                print("Meet error but I will try again after 2s. Reason:", e)
                print(f"Retry after 2s.")
                time.sleep(2)
            self.final_table = table
        def explain_misdetected_object(text):
            contents = [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": text
                        },
                        {
                            "text": "You need to make your answer compact and clear. Your output will be added into the a comprehensive analysis with other answers and then sent to the final integrator model."
                        }
                    ]
                }
            ]
            try:
                while True:
                    response = self.misdetected_object_assistant_model.generate_content(
                        contents=contents,
                        generation_config=self.generation_config,
                        safety_settings=self.safety_settings
                    )
                    break
            except KeyboardInterrupt:
                print("Keyboard interrupt received, exiting...")
                raise
            except Exception as e:
                print("Meet error but I will try again after 2s. Reason:", e)
                print(f"Retry after 2s.")
                time.sleep(2)
            objects = response.text
            contents = [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": "This is the table, you can find the visible area of each image here:\n" + self.final_table
                        },
                        {
                            "text": "This is the object(s) you need to check: " + objects
                        },
                        {
                            "text": "You need to make your answer compact and clear. Your output will be added into the a comprehensive analysis with other answers and then sent to the final integrator model."
                        }
                    ]
                }
            ]
            try:
                while True:
                    response = self.misdetected_object_assistant_model.generate_content(
                        contents=contents,
                        generation_config=self.generation_config,
                        safety_settings=self.safety_settings
                    )
                    break
            except KeyboardInterrupt:
                print("Keyboard interrupt received, exiting...")
                raise
            except Exception as e:
                print("Meet error but I will try again after 2s. Reason:", e)
                print(f"Retry after 2s.")
                time.sleep(2)
            return response.text
        def table_integration(text):
            contents = [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": text
                        },
                        {
                            "text": "You need to make your answer compact and clear. Your output will be added into the a comprehensive analysis with other answers and then sent to the final integrator model."
                        }
                    ]
                }
            ]
            try:
                while True:
                    response = self.table_integration_model.generate_content(
                        contents=contents,
                        generation_config=self.generation_config,
                        safety_settings=self.safety_settings
                    )
                    table = response.text
                    break
            except KeyboardInterrupt:
                print("Keyboard interrupt received, exiting...")
                raise
            except Exception as e:
                print("Meet error but I will try again after 2s. Reason:", e)
                print(f"Retry after 2s.")
                time.sleep(2)
            # print(table)
            checker_stp1 = threading.Thread(target=inconsistencies_checker, args=(table,))
            checker_stp2 = threading.Thread(target=mislabeled_object_checker, args=(table,))
            checker_stp3 = threading.Thread(target=misdetected_object_checker, args=(table,))
            checker_stp1.start()
            checker_stp2.start()
            checker_stp3.start()
            checker_stp1.join()
            checker_stp2.join()
            checker_stp3.join()

            renew_table(table)
            return self.final_table
        stp1 = threading.Thread(target=observation_range, args=(text,))
        stp2 = threading.Thread(target=global_coordinate, args=(text,))
        stp1.start()
        stp2.start()
        stp1.join()
        stp2.join()
        table = table_integration(self.text2table)
        # print(self.error_checker + table)
        contents = [
            {
                "role": "user",
                "parts": [
                    {
                        "text": table 
                    },
                    {
                        "text": "Please do not add any prefix or suffix."
                    }
                ]
            }
        ]
        try:
            while True:
                response = self.global_map_analysis_model.generate_content(
                    contents=contents,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings
                )
                self.global_map = response.text
                break
        except KeyboardInterrupt:
            print("Keyboard interrupt received, exiting...")
            raise
        except Exception as e:
            print("Meet error but I will try again after 2s. Reason:", e)
            print(f"Retry after 2s.")
            time.sleep(2)
        return self.global_map
        
    
    def map_updater(self, text, cord):
        local_info = 'When the world coordinates of the central of image is: ' + cord + '\nThe local coordinate of objects in the image is:\n' + text + '\n'
        contents = [
            {
                "role": "user",
                "parts": [
                    {
                        "text": "This is the original global map:\n" + self.global_map + "\nThis is the newest local map information:\n" + local_info + "\nPlease update the global map with this new local map and output the updated global map. Do not add any prefix or suffix."
                    }
                ]
            }
        ]
        try:
            while True:
                response = self.updator_model.generate_content(
                    contents=contents,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings
                )
                self.global_map = response.text
                break
        except KeyboardInterrupt:
            print("Keyboard interrupt received, exiting...")
            raise
        except Exception as e:
            print("Meet error but I will try again after 2s. Reason:", e)
            print(f"Retry after 2s.")
            time.sleep(2)
        return response.text, local_info
    
    def map_getter(self):
        return self.global_map  
    


if __name__ == "__main__":
    client = GeminiReasoner()
    import time
    import rospy
    from std_msgs.msg import String
    from planners import gemini_planner
    import sys
    client_ = gemini_planner.GeminiPlanner()
    rospy.init_node('gemini_reasoner')
    global_map_pub = rospy.Publisher('/llm/global_map', String, queue_size=10)
    global_map_data = String()
    # task = "assemle the word LOVE, from top to bottom"
    # response = client.global_planner(task)
    # print(response)
    # sys.exit()
    # def convert_string_to_list(string):
    #     if not string.startswith('[') and not string.endswith(']'):
    #         print("The input string is not in the correct list format.")
    #         return None

    #     try:
    #         string = string[1:-1]
            
    #         elements = string.split("; ")
            
    #         result = []
    #         for element in elements:
    #             element = element.strip() 
    #             if element.startswith("[") and "]: '" in element:
    #                 result.append(f"{element}")
            
    #         return result

    #     except Exception as e:
    #         print(f"Error converting string to list: {e}")
    #         return None
    # task_list = convert_string_to_list(response)
    # for task in task_list:
    #     response = client.mf_planner(task)
    #     print(response)
    #     print('--------------------------')

    text = 'When the world coordinates of the central of\
  \ image is: (4.8, 4.7).\nThe local coordinate of objects in the image is:\nfname:\
  \ Block_E, coordinate: x: -8.3, y: 5.2\nname: Block_O, coordinate: x: -5.1, y: -0.1\n\
  name: Block_V, coordinate: x: 0.5, y: 4.9\nname: Block_Unknown, coordinate: x: 1.0,\
  \ y: -6.3\nname: Block_Unknown, coordinate: x: -9.0, y: -6.0\nname: Robot_Dog, coordinate:\
  \ x: -4.5, y: -5.0\n\nWhen the world coordinates of the central of image is: (-4.9,\
  \ 4.2).\nThe local coordinate of objects in the image is:\nfname: Block_L, coordinate:\
  \ x: -4.3, y: -5.3\nname: Block_O, coordinate: x: 7.0, y: 0.3\nname: Block_E, coordinate:\
  \ x: 1.4, y: -6.0\nname: Block_Unknown, coordinate: x: 11.8, y: 4.0\nname: Block_Unknown,\
  \ coordinate: x: 11.8, y: -6.5\nname: Robot_Dog, coordinate: x: 6.5, y: -5.4\n\n\
  When the world coordinates of the central of image is: (-4.7, -1.7).\nThe local\
  \ coordinate of objects in the image is:\nfname: Block_L, coordinate: x: -4.3, y:\
  \ 1.8\nname: Block_B, coordinate: x: 1.3, y: 1.4\nname: Block_Unknown, coordinate:\
  \ x: 6.3, y: 6.8\nname: Block_V, coordinate: x: 10.5, y: -0.2\nname: Robot_Dog,\
  \ coordinate: x: 6.0, y: 2.1\n\nWhen the world coordinates of the central of image\
  \ is: (4.8, -1.4).\nThe local coordinate of objects in the image is:\nfname: Block_O,\
  \ coordinate: x: -5.7, y: 6.5\nname: Block_V, coordinate: x: 0.7, y: 0.2\nname:\
  \ Block_B, coordinate: x: -9.7, y: -0.2\nname: Robot_Dog, coordinate: x: -5.5, y:\
  \ 1.3\n\nWhen the world coordinates of the central of image is: (0.2, -0.6).\nThe\
  \ local coordinate of objects in the image is:\nfname: Block_O, coordinate: x: 0.2,\
  \ y: 6.5\nname: Block_L, coordinate: x: -10.3, y: 0.3\nname: Block_V, coordinate:\
  \ x: 5.7, y: -0.5\nname: Robot_Dog, coordinate: x: 0.0, y: 1.3'
    start_time = time.time()
    response = client.map_constructor(text)
    print(response)
    print(time.time() - start_time)
    # response_ = client_.classifier('move the I cube to the right of the L cube', data=response)
    # global_map_data.data = response_
    # global_map_pub.publish(global_map_data)
    # # print(response_)
    # start_time = time.time()
    response, local_info = client.map_updater("name: Block_U, coordinate: x: -0.2, y: 0.8\n\
                                    name: Robot_Dog, coordinate: x: 0.5, y: -2.9\n\n", "-6.5, 5.0")
    # response_ = client_.classifier('move the X cube to the right of the L cube', data=response)
    print(response)
    # # global_map_data.data = response_
    # # global_map_pub.publish(global_map_data)
    # print(time.time() - start_time)
    # response = client.map_getter()
    # print(response)