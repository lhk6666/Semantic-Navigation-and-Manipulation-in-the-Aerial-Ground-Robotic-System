import cv2
from cv_bridge import CvBridge, CvBridgeError
import base64
import random
import numpy as np
import time
import websockets
import math
import json

def convert_string_to_list(string):
    # Check if the input string starts with '[' and ends with ']', otherwise return None
    if not string.startswith('[') and not string.endswith(']'):
        print("The input string is not in the correct list format.")
        return None

    try:
        # Remove the surrounding brackets
        string = string[1:-1]
        
        # Split by ", " to separate elements
        elements = string.split("; ")
        
        # Clean up each element: remove additional symbols and make it into "Device/Action" format
        result = []
        for element in elements:
            element = element.strip()  # Remove any surrounding whitespace
            if element.startswith("[") and "]: '" in element:
                # Split by "]: '" to get the device and the action
                # device, action = element.split("]: '")
                # Replace spaces in the device name with underscores and combine the parts
                result.append(f"{element}")
        
        return result

    except Exception as e:
        print(f"Error converting string to list: {e}")
        return None

def draw_grid_with_border(image, grid_size=11, border_thickness=2, propotion=1):
    # Get image dimensions
    height, width = image.shape[:2]
    add_witdth = 100 
    add_height = 50 
    step_x = int(116 / propotion)
    step_y = int(102 / propotion)
    
    # Calculate the step size for the grid
    grid_size_x = width // step_x
    grid_size_y = height // step_y
    
    # Create a new image with extra space for the border
    bordered_image = cv2.copyMakeBorder(
        image, border_thickness, border_thickness + add_height, border_thickness + add_witdth, border_thickness,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    # Get new dimensions after adding the border
    bordered_height, bordered_width = bordered_image.shape[:2]

    # Draw vertical grid lines
    for i in range(1, grid_size_x + 1):
        x = i * step_x + border_thickness
        cv2.line(bordered_image, (x + add_witdth, border_thickness), (x + add_witdth, bordered_height - border_thickness), (255, 255, 255), 2)
        # Add column numbers at the top
        cv2.putText(bordered_image, str(float(i-(grid_size_x+1)//2)), (x + grid_size_x//2, bordered_height - add_height//2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw horizontal grid lines
    for i in range(1, grid_size_y + 1):
        y = i * step_y + border_thickness
        cv2.line(bordered_image, (border_thickness, y), (bordered_width - border_thickness, y), (255, 255, 255), 2)
        # Add row numbers at the left
        cv2.putText(bordered_image, str((grid_size_y+1)/2-i), (10, y - step_y // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Draw the outer border
    cv2.rectangle(bordered_image, (border_thickness + add_witdth, border_thickness), 
                (bordered_width - border_thickness, bordered_height - border_thickness), 
                (255, 255, 255), border_thickness)

    return bordered_image

def to_base64(msg, compressed=False, enhanced=False, save=False, show=False):
    bridge = CvBridge()
    dim = (int(1280/2), int(720/2))
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        if enhanced:
            h, w = cv_image.shape[:2]
            x1, y1 = w // 4, h // 4
            x2, y2 = 3 * w // 4, 3 * h // 4
            cropped_image = cv_image[y1:y2, x1:x2]
            cv_image = cv2.resize(cropped_image, (w, h), interpolation=cv2.INTER_LINEAR)
            grid_image = draw_grid_with_border(cv_image, propotion= 0.5)
        else:
            grid_image = draw_grid_with_border(cv_image)
        if compressed:
            grid_image = cv2.resize(grid_image, dim, interpolation=cv2.INTER_AREA)
            cv_image = cv2.resize(cv_image, dim, interpolation=cv2.INTER_AREA)
        _, buffer = cv2.imencode('.jpg', grid_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        _, buffer = cv2.imencode('.jpg', cv_image)
        img_base64_norm = base64.b64encode(buffer).decode('utf-8')

        if save:
            random_number = random.uniform(0, 1000)
            cv2.imwrite(f'/home/dragon_llm/ros/llm_ws/src/image_saver/grid_image_{random_number}.jpg', grid_image)
            random_number = random.uniform(0, 1000)
            cv2.imwrite(f'/home/dragon_llm/ros/llm_ws/src/image_saver/normal_image_{random_number}.jpg', cv_image)
        if show:
            cv2.imshow("Image with Grid", grid_image)
            cv2.waitKey(1)

    except CvBridgeError or KeyboardInterrupt as e:
        raise ValueError("Failed to read image")
    return img_base64, img_base64_norm

def remove_key_with_braces(json_data):
    start_index = json_data.find("[")
    end_index = json_data.rfind("]")

    cleaned_json_data = json_data[start_index:end_index + 1]

    return cleaned_json_data

def prize_calculator(sum_prize, gpt_output):
    sum_prize += gpt_output.usage.prompt_tokens * 0.00000250
    sum_prize += gpt_output.usage.completion_tokens * 0.00001000

    return sum_prize

def interpolate_position(current_position, target_position, factor=0.5):
    return np.array(current_position) + factor * (np.array(target_position) - np.array(current_position))

def is_valid_jpeg(bytes_data):
    if len(bytes_data) < 4:  
        return False

    return (bytes_data[0] == 0xFF and
            bytes_data[1] == 0xD8 and
            bytes_data[-2] == 0xFF and
            bytes_data[-1] == 0xD9)
    
def HsV(image, value=15):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    s = cv2.add(s, value)  

    hsv_enhanced = cv2.merge([h, s, v])

    enhanced_image = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
    
    return enhanced_image

async def request_from_address(to_base64 = False):
    camera_ip = "192.168.1.30"  
    port = 8000
    websocket_url = f"ws://{camera_ip}:{port}"
    img_base64 = None
    img_base64_norm = None
    grid_image = None
    
    bridge = CvBridge()

    async with websockets.connect(websocket_url) as websocket:
        # print("Connected to WebSocket server.")

        while 1: 
            try:
                image_data = await websocket.recv()

                image_data_np = np.frombuffer(image_data, dtype=np.uint8)
                if not is_valid_jpeg(image_data_np):
                    # print("Invalid JPEG data received. Retrying...")
                    continue

                image_cv = cv2.imdecode(image_data_np, cv2.IMREAD_COLOR)
                # image_cv = HsV(image_cv, value=15)
                
                ros_image = bridge.cv2_to_imgmsg(image_cv, encoding="bgr8")

                if to_base64:
                    grid_image = draw_grid_with_border(image_cv)
                    _, buffer_grid = cv2.imencode('.jpg', grid_image)
                    img_base64 = base64.b64encode(buffer_grid).decode('utf-8')

                # _, buffer_norm = cv2.imencode('.jpg', image_cv)
                # img_base64_norm = base64.b64encode(buffer_norm).decode('utf-8')

                # print("Image processed successfully.")
                break

            except Exception as e:
                print(f"Error during image processing: {e}")
                time.sleep(0.1) 

    if grid_image is not None:
        cv2.imwrite('test_image.jpg', grid_image)
    if to_base64:
        return img_base64, img_base64_norm
    else:
        return ros_image

def angle_greater_than_90(u, v):
    u = np.array(u, dtype=float)
    v = np.array(v, dtype=float)

    dot_product = np.dot(u, v)
    
    return dot_product <= 1e-6

def extract_coordinates_by_type(data, object_type, threshold=2):
    coords_list = []     
    carry_flag = False
    carry_success = False
    coord_head = None      

    def get_coord(item):
        coord = item.get("coordinates") or item.get("coordinate")
        return coord

    for obj in data:
        if obj.get("type") == object_type:
            if obj.get("name") == "Robot_Dog":
                for part in obj.get("parts", []):
                    if part.get("part") == "head":
                        head_coord = part.get("coordinates") or part.get("coordinate")
                        if head_coord:
                            coord_head = [head_coord.get("x", 0), head_coord.get("y", 0)]
                        break

            coord = get_coord(obj)
            
            if coord:
                x = coord.get("x", 0) 
                y = coord.get("y", 0) 
            else:
                if obj.get("name") == "Robot_Dog":
                    return None, carry_flag, carry_success
                else:
                    if object_type == "main":
                        carry_flag = True
                continue
            if object_type != "main":
                    coords_list.append([x, y])
            else:
                if obj.get("name") == "Robot_Dog":
                    coords_list.insert(0, [x, y])
                else:
                    carry_flag = True
                    coords_list.append([x, y])

    if object_type == "obstacle":
        for obj in data:
            if obj.get("type") == "landmark":
                coord = get_coord(obj)
                if coord:
                    x = coord.get("x", 0)
                    y = coord.get("y", 0)
                    coords_list.append([x, y])
    
    if object_type == "target" and not coords_list:
        for obj in data:
            if obj.get("type") == "landmark":
                coord = get_coord(obj)
                if coord:
                    x = coord.get("x", 0)
                    y = coord.get("y", 0)
                    direction = obj.get("direction")
                    if direction == "left":
                        coords_list.append([x - 4, y])
                    elif direction == "right":
                        coords_list.append([x + 4, y])
                    elif direction == "front":
                        coords_list.append([x, y + 4])
                    elif direction == "back":
                        coords_list.append([x, y - 4])

    if object_type == "main":
        if carry_flag:
            if len(coords_list) > 1:
                dog_orientation = extract_orientation_by_parts(data)
                x1, y1 = coords_list[0]
                x2, y2 = coords_list[1]
                dog2obj_orientation = math.atan2(y2 - y1, x2 - x1)
                if coord_head:
                    head2obj_dist = math.sqrt((x2 - coord_head[0])**2 + (y2 - coord_head[1])**2)
                else:
                    head2obj_dist = 0
                diff_orientation = (dog_orientation - dog2obj_orientation + math.pi) % (2 * math.pi) - math.pi
                if abs(diff_orientation) < 1 and head2obj_dist < 1.5 * threshold:
                    carry_success = True
            elif len(coords_list) == 1:
                carry_success = True
        return coords_list, carry_flag, carry_success

    elif object_type == "obstacle":
        return coords_list if coords_list else []

    else:
        return coords_list[0] if coords_list else None


def extract_orientation_by_parts(data):
    def calculate_orientation(parts_coords):
        if "head" in parts_coords and "body" in parts_coords and "tail" in parts_coords:
            head = np.array(parts_coords["head"])
            body = np.array(parts_coords["body"])
            tail = np.array(parts_coords["tail"])
            # Compute vector from tail to body and from body to head
            vec1 = body - tail
            vec2 = head - body
            # Average the two vectors to get a more robust orientation
            avg_vec = (vec1 + vec2) / 2
            return math.atan2(avg_vec[1], avg_vec[0])
        return 0.0

    if not isinstance(data, list):
        raise ValueError("Input data should be a list.")

    # objects = data.get("Objects", [])
    objects = data
    for obj in objects:
        if obj['name'] == "Robot_Dog":
            try:
                parts = obj['parts']
                parts_coords = {}
                for part in parts:
                    if part['part'] is None:
                        continue
                    elif part['part'] == "head":
                        try:
                            head_coord = part['coordinates']
                        except:
                            head_coord = part['coordinate']
                        if head_coord is not None:
                            parts_coords["head"] = [head_coord["x"], head_coord["y"]]
                    elif part['part'] == "body":
                        try:
                            body_coord = part['coordinates']
                        except:
                            body_coord = part['coordinate']
                        if body_coord is not None:
                            parts_coords["body"] = [body_coord["x"], body_coord["y"]]
                    elif part['part'] == "tail":
                        try:
                            tail_coord = part['coordinates']
                        except:
                            tail_coord = part['coordinate']
                        if tail_coord is not None:
                            parts_coords["tail"] = [tail_coord["x"], tail_coord["y"]]
            except:
                return 0.0
            orientation = calculate_orientation(parts_coords)
            return orientation
    
    return 0.0

def auto_correct_json(json_string):
    """
    Attempts to auto-correct common JSON format issues like:
    - Missing or extra brackets
    - Trailing commas

    Args:
        json_string (str): The input JSON string to be corrected.

    Returns:
        str: The corrected JSON string.
        None: If the input cannot be corrected.
    """
    try:
        # Attempt to load the JSON string directly
        return json.dumps(json.loads(json_string), indent=4)
    except json.JSONDecodeError as e:
        print("Initial JSON parsing failed, attempting auto-correction...")

    # Attempt to fix trailing commas
    corrected = json_string

    # Remove trailing commas (before a closing bracket or brace)
    import re
    corrected = re.sub(r',\s*([}\]])', r'\1', corrected)

    # Ensure proper bracket closure
    open_brackets = corrected.count("[")
    close_brackets = corrected.count("]")
    open_braces = corrected.count("{")
    close_braces = corrected.count("}")

    if open_brackets > close_brackets:
        corrected += "]" * (open_brackets - close_brackets)
    elif close_brackets > open_brackets:
        corrected = corrected.rstrip("]")

    if open_braces > close_braces:
        corrected += "}" * (open_braces - close_braces)
    elif close_braces > open_braces:
        corrected = corrected.rstrip("}")

    # Try parsing again after correction
    try:
        return json.dumps(json.loads(corrected), indent=4)
    except json.JSONDecodeError:
        print("Auto-correction failed. Please check the input JSON manually.")
        return None


import numpy as np

class KalmanFilter:
    def __init__(self, process_variance, measurement_variance, initial_estimate=0, initial_error_variance=1):
        self.initial_estimate = initial_estimate
        self.initial_error_variance = initial_error_variance
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.reset()

    def reset(self):
        self.estimate = self.initial_estimate
        self.error_variance = self.initial_error_variance
        self.kalman_gain = 0

    def update(self, measurement):
        self.error_variance += self.process_variance

        self.kalman_gain = self.error_variance / (self.error_variance + self.measurement_variance)
        self.estimate += self.kalman_gain * (measurement - self.estimate)
        self.error_variance *= (1 - self.kalman_gain)

        return self.estimate

# if __name__ == "__main__":
#     process_variance = 1e-4  
#     measurement_variance = 1e-2  

#     kalman_filter = KalmanFilter(process_variance, measurement_variance)

#     sensor_data = [45, 46, 48, 47, 49, 48, 50, 48.5, 49.5, 51, 48, 45, 43, 20, 44, 42,41,40,40,40] 
#     filtered_data = []

#     for measurement in sensor_data:
#         filtered_value = kalman_filter.update(measurement)
#         filtered_data.append(filtered_value)
#         print(f"测量值: {measurement:.2f}, 滤波值: {filtered_value:.2f}")

#     import matplotlib.pyplot as plt

#     plt.plot(sensor_data, label="Raw Data")
#     plt.plot(filtered_data, label="Kalman Filtered Data")
#     plt.xlabel("Time")
#     plt.ylabel("Value")
#     plt.legend()
#     plt.title("Kalman Filter Example")
#     plt.show()

#Debug 
if __name__== "__main__":
    data = [
        {
            "name": "Robot_Dog",
            "type": "main",
            "parts": [
            {
                "part": "head",
                "coordinate": {
                "x": 0.2,
                "y": 1.2
                }
            },
            {
                "part": "body",
                "coordinate": {
                "x": -0.8,
                "y": 2.0
                }
            },
            {
                "part": "tail",
                "coordinate": {
                "x": -1.3,
                "y": 2.4
                }
            }
            ],
            "coordinate": {
            "x": -0.8,
            "y": 2.0
            }
        },
        {
            "name": "X_Cube",
            "type": "main",
            "coordinate": None
        },
        {
            "name": "L_Cube",
            "type": "landmark",
            "direction": "right",
            "coordinate": {
            "x": -0.7,
            "y": -1.1
            }
        },
        {
            "name": "Block_I",
            "type": "obstacle",
            "coordinate": {
            "x": -8.5,
            "y": -5.3
            }
        }
    ]
    # data = json.load(data)
    # body_position, carry_flag, carry_success = extract_coordinates_by_type(data, 'target')
    # print(body_position, carry_flag, carry_success)
    target = extract_coordinates_by_type(data, 'target')
    print(target)