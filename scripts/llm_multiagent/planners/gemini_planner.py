import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
import numpy as np
import base64
import matplotlib.pyplot as plt
import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting
import time
import json
import tool
import yaml
from pathlib import Path

class GeminiPlanner:
    def __init__(self):
        cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        vertexai.init(project=cfg["project_id"],
                        location=cfg["finetuned_location"],)
        finetuned_model_id = "projects/" + str(cfg["project_id"]) + "/locations/" + cfg["finetuned_location"] + "/endpoints/" + str(cfg["finetuned_model"])
        self.describe_model = GenerativeModel(
            finetuned_model_id,
            system_instruction= "You are an environment describer. Your task is to describe the environment to the user with object names and coordinates in JSON format.\n\
                                The environment is labelled with 23 * 13 coordinate point, with the origin at central. The x-axis extends from -11 to 11 (left to right), and the y-axis extends from -6 to 6 (bottom to top).  The distance for each adjacent coordinate (left, right, front, back) point is 1 unit (80 pixels)\n\
                                Answer should in JSON format.\n\
                                If the taks is description only, the format of JSON could be Objects: name; (parts - only robot dog have parts, you have to add parts if there is robot dog); coordinate: x, y.\n\
                                Follow these steps:\n\
                                1. Analyze the environment visually or textually.\n\
                                2. Identify all objects present in the environment, but do not include the objects you did not see.\n\
                                3. For each object:\n\
                                    * Determine its name.\n\
                                    * If the object is the robot dog, you must identify its parts (head, tail, body) and their coordinates. To recognize its part, the robot dog is primarily silver, with a black head, a red tail, and a thin black belt around its body. Its length is approximately 2 units. **If you can not see the robot, you can ouput its coordinate as 'null'**\n\
                                    * Determine the object's overall coordinate, focusing on points around the target object or the robot dog's parts. Do not rely on the original resolution coordinates.\n\
                                4. Construct the JSON output as described above. Ensure all objects are included in the JSON array.\n\
                                Notice:\n\
                                * The letter sometimes are rotated, you need to carefully recognize it as the correct letter.\n\
                                * If you can not make sure about exact name of the object, you can output it like - 'name': 'xxx_Unknown'.\n\
                                * Never use single quota, use double quota"
                                
        )
        self.classify_model = GenerativeModel(
            cfg["flash_model"],
            system_instruction= "You are an environment classifier. Your task is to classify objects according to the task content.\n\
                                Answer should in JSON format.\n\
                                If the task is classification according to the task, the format of JSON could be Objects: 'type': select from the four options - main, target, landmark, and obstacle; ('direction' - only when the object is landmark in the task); ('parts' - only robot dog have parts, every parts have their own coordinates); 'coordinate': 'x', 'y'. Notice: Do not include target and main in the obstacles. The main coordinate of the robot dog can represent by its body.\n\
                                Follow these steps:\n\
                                1. Identify all objects present in the environment and task textually.\n\
                                2. For each object:\n\
                                    * Determine their name.\n\
                                    * If the object is not in the environment but mentioned in the task, you need to add it in the Objects of your output like the format of others, but because you can not find its coordinate, leave its coordinate like - 'coordinate': null. Notice: Block and cube are same things\n\
                                    * Classify the object based on the task content.\n\
                                    * The 'main' is typically the robot dog, unless it is carrying something, in which case the carried object is the main. \n\
                                    * The 'target' is the object the main is trying to reach or interact with.\n\
                                    * Sometimes, the task will mention of xx side of xx, so the object should be 'landmark' but not 'target' cause its position is not the destination. For this case, add direction as a separate field alongside type, name, and coordinate, to specify the direction relative to the landmark. Notice, 'target' and 'landmark' can only exist one in your output, cause they are same level in one task. Direction includes 'left', 'right', 'up' and 'down'\n\
                                    * All other objects that could hinder the task are classified as 'obstacle'. Do not classify the target or main as obstacle.\n\
                                3. Construct the JSON output as described above. Ensure all objects are included in the JSON array.\n\n\
                                Example:\n\
                                * If the task is for the robot dog to retrieve a ball, the ball would be the 'target', the robot dog is the 'main', and any other objects in the environment would be 'obstacle'.\n\
                                * If the robot dog is carrying a box to the table, the box and the robot dog would be the 'main' and the table would be the 'target'.\n\
                                * If the robot move to the right side of the table, the table's 'type' would be the 'landmark' and the 'direction' would be 'right'.\n\
                                * If the task is ask the robot dog to the right side of blue desk, but you can not find the blue desk in the environment, you need to output it like  'name': 'Blue_Desk', 'type': 'landmark', 'direction': 'right', 'coordinate': null.\n\
                                * If the task is ask you move or carry something to (xx, xx), because the coordinate is for absolute, you can output the 'name' as 'Target' and use coordinate provided in the sentence\n"
        )
        self.classify_fast_model = GenerativeModel(
            finetuned_model_id,
            system_instruction = "You are an environment classifier. Your task is to classify objects according to the task content.\n\
                                The environment is labelled with 23 * 13 coordinate point, with the origin at central. The x-axis extends from -11 to 11 (left to right), and the y-axis extends from -6 to 6 (bottom to top).  The distance for each adjacent coordinate point (left, right, front, back) is 1 unit (80 pixels for 1 unit)\n\
                                Answer should in JSON format.\n\
                                If the task is classification according to the task, the format of JSON could be Objects: 'type': select from the four options - main, target, landmark, and obstacle; ('direction' - it is the parameter only for 'landmark'. Four direction can be used -  **'left'**, **'right'**, **'front'** and **'back'**); ('parts' - only robot dog have parts, every parts have their own coordinates); 'coordinate': 'x', 'y'. Notice: Do not include target and main in the obstacles. The main coordinate of the robot dog can represent by its body.\n\
                                Follow these steps:\n\
                                1. Identify all objects present in the environment and task.\n\
                                2. For each object:\n\
                                    * Determine their name.\n\
                                    * If the object is not in the environment but mentioned in the task, you need to add it in your output like the format of others, but because you can not find its coordinate, leave its coordinate like - 'coordinate': null. Notice: Block and cube are same things\n\
                                    * Classify the object based on the task content.\n\
                                    * The 'main' is typically the robot dog, unless it is carrying something, in which case the carried object and robot are the main. \n\
                                    * The 'target' is the object the main is trying to reach or interact with.\n\
                                    * Sometimes, the task will mention of xx side of xx, so the object should be 'landmark' but not 'target' cause its position is not the destination. For this case, add direction as a separate field alongside type, name, and coordinate, to specify the direction relative to the landmark. Notice, 'target' and 'landmark' can only exist one in your output, cause they are same level in one task. Direction includes **'left'**, **'right'**, **'front'** and **'back'**\n\
                                    * All other objects that could hinder the task are classified as 'obstacle'. Do not classify the target or main as obstacle.\n\
                                    * Sometimes the task will mention the object which is not in the environment, you need to add it in your output like the format of others, but because you can not find its coordinate, leave its coordinate like - **'coordinate': null**.\n\
                                    * Sometimes the task will mention the robot dog move or carry something to (xx, xx), because the coordinate is for absolute, you **have to overlook it**, you **must** output 'null' or 'x: 0, y: 0' in the coordinate to let the image's to lead the dog to the destination.\n\
                                3. Construct the JSON output as described above. Ensure all objects are included in the JSON array.\n\n\
                                Example:\n\
                                * If the task is for the robot dog to retrieve a ball, the ball would be the 'target', the robot dog is the 'main', and any other objects in the environment would be 'obstacle'.\n\
                                * If the robot dog is carrying a box to the table, the box and the robot dog would be the 'main' and the table would be the 'target'.\n\
                                * If the robot move to the right side of the table, the table's 'type' would be the 'landmark' and the 'direction' would be 'right'.\n\
                                * If the task is ask the robot dog to the back side of blue desk, but you can not find the blue desk in the environment, you need to output it like  'name': 'Blue_Desk', 'type': 'landmark', 'direction': 'back', 'coordinate': null.\n\n\
                                * If the task is ask the robot dog move or carry something to (xx, xx), because the coordinate is for absolute, you **have to overlook it**, you **must** output 'null' or 'x: 0, y: 0' in the coordinate to let the image's to lead the dog to the destination.\n\
                                Notice:\n\
                                * If the object is not in the environment but mentioned in the task, you need to add it in your output like the format of others, but because you can not find its coordinate, leave its coordinate like - 'coordinate': null.\n\
                                * Cube and block are the same thing.\n\
                                * The robot dog is primarily silver, with a black head, a red tail, and a thin black belt around its body. Its length is approximately 2 units. You need to identify its parts (head, tail, body) and their coordinates. **If you can not see the robot, you can ouput its coordinate as 'null'**\n\
                                * If the task mentioned a absoult coordinate, you can not directly use the number, because the coordinate in the image is different from the original resolution coordinates. You **must** output 'null' in the coordinate.\n\
                                * If you can not make sure about exact name of the object, you can output it like - 'name': 'xxx_Unknown'.\n\
                                * The letter sometimes are rotated, you need to carefully recognize it as the correct letter.\n\
                                * Do not let the task effect your recognition of the object in the environment.\n"
        )
        self.generation_config = {
                    "max_output_tokens": 8192,
                    "temperature": 0.5,
                    "top_p": 0.5,
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

    def draw_elements(self, potential_frame_, x_min = -11, x_max = 11, y_min = -6, y_max = 6, resolution = 80):
        # Draw grid and labels
        potential_frame = potential_frame_.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        height, width, _ = potential_frame.shape
        if height == 1080 and width == 1920:
            # print("The image is 1080p.")
            for x in np.arange(x_min, x_max + 1, 1):
                for y in np.arange(y_min, y_max + 1, 1):
                    col = int((x - x_min) * resolution)
                    row = int((y_max - y) * resolution)
                    surrounding_area = potential_frame[row + 50:row + 60, col + 52:col + 120]
                    brightness = np.mean(surrounding_area)
                    font_color = (255, 255, 255) if brightness < 127 else (0, 0, 0)
                    cv2.circle(potential_frame, (col + 80, row + 60), 3, font_color, -1)
                    if int(x) < 0:
                        if int(x) < -9:
                            if int(y) >= 0:
                                cv2.putText(potential_frame, f'({int(x)}, {int(y)})', (col + 40, row + 75), font, font_scale, font_color, thickness, cv2.LINE_AA)
                            else:
                                if int(x) < -10:
                                    cv2.putText(potential_frame, f'({int(x)}, {int(y)})', (col + 32, row + 75), font, font_scale, font_color, thickness, cv2.LINE_AA)
                                else:
                                    cv2.putText(potential_frame, f'({int(x)}, {int(y)})', (col + 35, row + 75), font, font_scale, font_color, thickness, cv2.LINE_AA)
                        else:
                            cv2.putText(potential_frame, f'({int(x)}, {int(y)})', (col + 45, row + 75), font, font_scale, font_color, thickness, cv2.LINE_AA)
                    else:
                        cv2.putText(potential_frame, f'({int(x)}, {int(y)})', (col + 56, row + 75), font, font_scale, font_color, thickness, cv2.LINE_AA)
        elif height == 720 and width == 1280:
            # print("The image is 720p.")
            x_min = -7
            x_max = 7
            y_min = -4
            y_max = 4
            for x in np.arange(x_min, x_max + 1, 1):
                for y in np.arange(y_min, y_max + 1, 1):
                    col = int((x - x_min) * resolution)
                    row = int((y_max - y) * resolution)
                    surrounding_area = potential_frame[row + 50:row + 60, col + 52:col + 120]
                    brightness = np.mean(surrounding_area)
                    font_color = (255, 255, 255) if brightness < 127 else (0, 0, 0)
                    cv2.circle(potential_frame, (col + 80, row + 40), 3, font_color, -1)
                    if int(x) < 0:
                        cv2.putText(potential_frame, f'({int(x)}, {int(y)})', (col + 45, row + 55), font, font_scale, font_color, thickness, cv2.LINE_AA)
                    else:
                        cv2.putText(potential_frame, f'({int(x)}, {int(y)})', (col + 56, row + 55), font, font_scale, font_color, thickness, cv2.LINE_AA)
        return potential_frame

    def describer(self, base64_img, json_format = True):
        contents = [
            {
                    "role": "user",
                    "parts": [
                        {
                            "inline_data": { 
                                "mime_type": "image/jpeg",
                                "data": base64_img
                            }
                        },
                        {
                            "text": "This is the image of the environment."
                        }
                    ]
            },
            {
                "role": "user",
                "parts": [
                    {
                        "text": "The json format should be [{'name': 'xxx', 'coordinate': {'x': x, 'y': y} }, ...]. Do not include the information in 'objects'."
                    },
                    {
                        "text": "Block and cube are the same things. You can use either of them. But do not use both."
                    },
                    {
                        "text": "**Never use single quotes** in the JSON format. You **must use double quotes**."
                    },
                    {
                        "text": "If you see color cube, do not describe it as Block_B, Block_R, Block_G, Block_Y. You should describe the full color name like Block_Blue, Block_Red, Block_Green, Block_Yellow."
                    },
                    {
                        "text": "Please describe the environment."
                    }
                ]
            }
        ]
        try:
            while True:
                response = self.describe_model.generate_content(
                    contents=contents,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings
                )
                buffer=response.text.replace("'", '"')
                if not json_format:
                    nonjson = self.json_extractor(buffer)
                break
        except KeyboardInterrupt:
            print("Keyboard interrupt received, exiting...")
            raise
        except Exception as e:
            print("Meet error but I will try again after 1s. Reason:", e)
            print(f"Retry after 1s.")
            time.sleep(1)
        if json_format:
            return response.text
        else:
            return response.text, nonjson
    
    def classifier(self, task, base64_img = None, data = None):
        if base64_img is not None:
            data = self.describer(base64_img)
        contents = [
            {
                "role": "user",
                "parts": [
                    {
                        "text": "This is the textual description of the environment."
                    },
                    {
                        "text": data
                    }
                ]
            },
            {
                "role": "user",
                "parts": [
                    {
                        "text": "Your Json format should use double quotes, not single quotes."
                    },
                    {
                        "text": f"The task is {task}. Please classify the objects in the environment and task. Do not add 'objects' before the JSON format."
                    }
                ]
            }
        ]
        try:
            while True:
                response = self.classify_model.generate_content(
                    contents=contents,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings
                )
                buffer=response.text.replace("'", '"')
                break
        except KeyboardInterrupt:
            print("Keyboard interrupt received, exiting...")
            raise
        except Exception as e:
            print("Meet error but I will try again after 2s. Reason:", e)
            print(f"Retry after 2s.")
            time.sleep(2)
        response = tool.remove_key_with_braces(buffer)
        return response
    
    def classifier_fast(self, task, base64_img):
        contents = [
            {
                    "role": "user",
                    "parts": [
                        {
                            "inline_data": { 
                                "mime_type": "image/jpeg",
                                "data": base64_img
                            }
                        },
                        {
                            "text": "This is the image of the environment."
                        }
                    ]
            },
            {
                "role": "user",
                "parts": [
                    {
                        "text": "You can not direct use the coordinate provided in task, because the coordinate in the image is local but the coordinate mentioned in the task is absolute. You must output 'null' or 'x: 0, y: 0' in the coordinate to let the image's to lead the dog to the destination."
                    },
                    {
                        "text": "If the robot dog is in the image, do not forget to add the 'parts' for the robot dog. Block and cube are the same things. You can use either of them. But do not use both."
                    },
                    {
                        "text": "If you see color cube, do not describe it as Block_B, Block_R, Block_G, Block_Y. You should describe the full color name like Block_Blue, Block_Red, Block_Green, Block_Yellow."
                    },
                    {
                        "text": "If the object is not in the environment but mentioned in the task, you need to add it in the Objects of your output like the format of others, but because you can not find its coordinate, leave its coordinate like - 'coordinate': null."
                    },
                    {
                        "text": f"The task is {task}. Please describe and classify the environment. If you can not see the object but mentioned in task, add it but leave the coordinate as null. Do not add 'objects' before the JSON format."
                    },
                ]
            }
        ]
        try:
            while True:
                response = self.classify_fast_model.generate_content(
                    contents=contents,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings
                )
                buffer=response.text.replace("'", '"')
                break
        except KeyboardInterrupt:
            print("Keyboard interrupt received, exiting...")
            raise
        except Exception as e:
            print("Meet error but I will try again after 2s. Reason:", e)
            print(f"Retry after 2s.")
            time.sleep(2)
        response = tool.remove_key_with_braces(buffer)
        return response
    
    def json_extractor(self, data):
        data_ = ""
        try:
            if data == "[]" or data == "":
                return "No Object in the environment."
            data = json.loads(data)
        except:
            return "No Object in the environment."
        for object in data:
            name = object['name']
            coordinate = object['coordinate']
            x = coordinate['x']
            y = coordinate['y']
            data_+= f"name: {name}, coordinate: x: {x}, y: {y}\n"
        return data_
    


if __name__ == "__main__":
    video_path = '/home/dragon_llm/ros/llm_ws/src/Storage_video/20250314-211003.mp4'
    # video_path = '/media/dragon_llm/3C09549315E08290/experiment record/video/20250227-112630.mp4'
    cap = cv2.VideoCapture(video_path)
    client = GeminiPlanner()
    import rospy
    from std_msgs.msg import String
    rospy.init_node('local_map_publisher', anonymous=True)
    local_map_pub = rospy.Publisher('/llm/local_map', String, queue_size=10)
    local_map_data = String()
        
    if not cap.isOpened():
        print("Error: Could not open video.")
    else:
        count = 900
        while True and not rospy.is_shutdown():
            ret, frame = cap.read()
            if not ret:
                break
            if count % 300 == 0:
                frame = client.draw_elements(frame)
                # frame = cv2.resize(frame, (1920, 1080))
                img_base64 = base64.b64encode(cv2.imencode('.jpeg', frame)[1]).decode()
                # Test describer
                # response = client.describer(img_base64)
                # print(response)
                
                response, nonjosn_response = client.describer(img_base64, json_format=False)
                print(nonjosn_response)

                # Test classifier
                # start_time = time.time()
                # response = client.classifier("Move to the C cube", img_base64)
                # print(response)
                # print(time.time()-start_time)

                start_time = time.time()
                response = client.classifier_fast("carry L cube to (4, 5)", img_base64)
                print(response)
                print(time.time()-start_time)
                local_map_data.data = response
                local_map_pub.publish(local_map_data)

                # Test classifier with text data
                # data = "```\nGlobal Map:\n- Block_L: (-6.8, 4.9)\n- Block_O: (6.95, 4.95)\n- Block_V: (3.52,\
                #      \ 3.18)\n- Block_E: (8.7, -3.25)\n- Robot_Dog: (0.2, -1.12)\n```"
                # response = client.classifier("Carry the L cube to (-5, 0)", data = data)
                # print(response)



                plt.figure(figsize=(10, 6))
                plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                plt.title("Test")
                plt.show()
            count += 1
        cap.release()