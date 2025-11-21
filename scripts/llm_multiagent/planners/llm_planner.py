import openai
import json
import matplotlib.pyplot as plt
import os
from openai import AzureOpenAI

class LLMPlanner:
    def __init__(self):
        # self.client = openai.OpenAI()
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
            api_version="2024-10-21",
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            )

    def remove_key_with_braces(self,json_data):
        start_index = json_data.find("[")
        end_index = json_data.rfind("]")

        cleaned_json_data = json_data[start_index:end_index + 1]

        return cleaned_json_data

    def init(self, data):
          self.data = data
          self.current_position = data["object position"]
          self.input = [
                {
                    "role": "user",
                    "content": (
                        "You task is to determine the movement path for an object on a 2D grid environment according to the environmet. You need to finally control the object to reach the target. Notice: The weight of avoid obstacles is ten times of the weight of reaching the target.\n"
                        "Constraints and rules:\n"
                        "1. Next position and current position must be between adjacent grid cells.\n"
                        "2. Sometime you can stop progressing towards the target, but focus on get away from the obstacles.\n"
                        "3. Use serval steps to bypass the obstacles is an good choice when you confusing.\n"
                        "4. Keep 2 grid units away from the obstacles is the best.\n"
                        '5. Give your answer in json format as follows: {"x": 1.0, "y": 1.0}\n\n'
                    )
                },
                {
                    "role": "user",
                    "content": "The start state of the environment is as follows" + json.dumps(data)
                },
                {
                    "role": "user",
                    "content": "Give me the path coordinates to reach the target in json format without any other extra explaination."
                },
            ]
          return input

    def run(self):
        response = self.client.chat.completions.create(
            model= 'gpt-4.1',
            messages=self.input,
            # temperature=0.2,
        )
        answer = response.choices[0].message.content
        # print(answer)
        # try:
        # answer = self.advisor(self.remove_key_with_braces(answer))
        # print(answer)
        answer = self.remove_key_with_braces(answer)
        print(answer)
        data = json.loads(answer)
        array = [[point['x'], point['y']] for point in data]
        return array
        # except Exception as e:
        #     print(e)
        #     return answer, np.array([0, 0])


if __name__ == "__main__":
    planner = LLMPlanner()
    data = {
        "object": "X cube",
        "object position": [0.0, 0.0],
        "obstacles position": [[-1.0, 1.0], [1.0, 1.0]],
        "target position": [0.0, 4.0]
    }
    planner.init(data)
    # Initialize plot
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-3.5, 3.5)
    ax.plot(*zip(*data["obstacles position"]), 'ro', label='Obstacles')
    ax.plot(*data["target position"], 'go', label='Target')
    object_position_plot, = ax.plot(*data["object position"], 'bo', label='Object')
    ax.legend()

    trajectory = [data["object position"]]


    array = planner.run()

    for point in array: 
        print("Next: ", point)
        trajectory.append(point)
        object_position_plot.set_data(*zip(*trajectory))
        plt.draw()
        plt.pause(0.5)

    plt.ioff()
    plt.show()
