import json
import os

import openai

# openai.api_key = os.getenv("OPENAI_API_KEY")


# function to create environment with LLM
def create_env(prompt):
    model_id = "ft:gpt-3.5-turbo-0613:webscience-lab::8DyQDcTj"

    response = openai.ChatCompletion.create(
        model=model_id,
        messages=[
            {
                "role": "system",
                "content": "The environment in reinforcement learning, where the agent progresses from left to right, is represented by a list of strings, one character per block. The following conditions are applied: '-' is a blank block, 'H' is a hard block, and 'S' is a soft block. The agent can walk on the 'H' and 'S' blocks and can exist in the '-' area. If there is no 'H' or 'S' block under the agent, it will fall. Please return a list that predicts what kind of environment it is from a prompt that describes the given environment. Please make all elements in the list have the same length. Also, only allow '-' , 'H', and 'S' characters in the elements. Please return with the specified character length. Do not output anything other than a list.",
            },
            {
                "role": "user",
                "content": "100*20 size Evolution Gym environment that is simple.",
            },
            {
                "role": "assistant",
                "content": "['----------------------------------------------------------------------------------------------------', '----------------------------------------------------------------------------------------------------', '----------------------------------------------------------------------------------------------------', '----------------------------------------------------------------------------------------------------', '----------------------------------------------------------------------------------------------------', '----------------------------------------------------------------------------------------------------', '----------------------------------------------------------------------------------------------------', '----------------------------------------------------------------------------------------------------', '----------------------------------------------------------------------------------------------------', '----------------------------------------------------------------------------------------------------', '----------------------------------------------------------------------------------------------------', '----------------------------------------------------------------------------------------------------', '----------------------------------------------------------------------------------------------------', 'HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH------------------------------------------------', '----------------------------------------------------HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH', '----------------------------------------------------------------------------------------------------', '----------------------------------------------------------------------------------------------------', '----------------------------------------------------------------------------------------------------', '----------------------------------------------------------------------------------------------------', '----------------------------------------------------------------------------------------------------']",
            },
            {"role": "user", "content": prompt},
        ],
    )
    env_list = eval(response["choices"][0]["message"]["content"])
    return env_list


def adjust_list(lst):
    # Initialize with 5 lines of hyphens
    adjusted_list = ["-" * 100 for _ in range(5)]
    for s in lst:
        s = "".join(c if c in "-HS" else "-" for c in s)

        if len(s) == 100:
            adjusted_list.append(s)
        elif len(s) < 100:
            adjusted_list.append(s + "-" * (100 - len(s)))
        else:
            adjusted_list.append(s[:100])
    # Extend with 7 lines of hyphens at the end
    adjusted_list.extend(["-" * 100 for _ in range(7)])
    return adjusted_list


def check_columns(lst):
    num_hyphens = 0
    for i in range(len(lst[0])):
        for j in range(len(lst)):
            if lst[j][i] != "-":
                num_hyphens = 0
                break
        else:
            num_hyphens += 1

        if num_hyphens >= 5:
            return False

    return True


def process_neighbour(i, j, grid, object_cells, object_type, processed_cells):
    if (
        i < 0
        or j < 0
        or i >= len(grid)
        or j >= len(grid[0])
        or grid[i][j] == "-"
        or processed_cells[i][j]
    ):
        return

    # セルを処理済みとマークする
    processed_cells[i][j] = True

    cell = (len(grid) - 1 - i) * len(grid[0]) + j
    object_cells.append(cell)
    object_type.append(grid[i][j])

    process_neighbour(i + 1, j, grid, object_cells, object_type, processed_cells)
    process_neighbour(i - 1, j, grid, object_cells, object_type, processed_cells)
    process_neighbour(i, j + 1, grid, object_cells, object_type, processed_cells)
    process_neighbour(i, j - 1, grid, object_cells, object_type, processed_cells)


def create_json_file(env_list):
    grid = [list(row) for row in env_list]
    objects_dict = dict()
    object_counter = 1

    grid_width = len(grid[0])
    grid_height = len(grid)

    # Initialize start_height with the minimum value
    start_height = 1  # Start from the bottom row (1)
    for i in range(grid_height):
        for j in range(min(6, grid_width)):  # Check only the first 5 columns
            if grid[i][j] == "H" or grid[i][j] == "S":
                start_height = max(start_height, grid_height - i)

    # 新しいグリッドを初期化 (すべてのセルがまだ処理されていないことを示す)
    processed_cells = [[False for _ in range(grid_width)] for _ in range(grid_height)]

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == "H" or grid[i][j] == "S":
                new_object_indices = []
                new_object_type = []

                # processed_cellsを渡す
                process_neighbour(
                    i, j, grid, new_object_indices, new_object_type, processed_cells
                )

                # Initialize new_object_types with 'S' as 2 and 'H' as 5
                new_object_types = [5 if t == "H" else 2 for t in new_object_type]

                # Modify the object type based on the requirements
                s_count = 0
                for k, t in enumerate(new_object_type):
                    if t == "S":
                        s_count += 1
                        if k == 0 or k == len(new_object_type) - 1 or s_count == 5:
                            new_object_types[k] = 5  # Convert to 'H'
                            s_count = 0

                new_object = {}
                new_object["indices"] = new_object_indices
                new_object["types"] = new_object_types
                new_object["neighbors"] = {
                    str(idx): [
                        n
                        for n in new_object_indices
                        if (
                            n // grid_width == idx // grid_width
                            and abs(n % grid_width - idx % grid_width) == 1
                        )
                        or (
                            abs(n // grid_width - idx // grid_width) == 1
                            and n % grid_width == idx % grid_width
                        )
                    ]
                    for idx in new_object_indices
                }

                objects_dict["new_object_" + str(object_counter)] = new_object
                object_counter += 1

                for idx in new_object_indices:
                    grid[(len(grid) - 1 - idx // grid_width)][
                        idx % grid_width
                    ] = "-"  # fixed here

    return_dict = {}
    return_dict["grid_width"] = grid_width
    return_dict["grid_height"] = grid_height
    return_dict["start_height"] = start_height
    return_dict["objects"] = objects_dict

    return return_dict


def generate_env(prompt):
    checked_list = False
    count = 0
    while not checked_list:
        count += 1
        if count > 10:
            break
        try:
            env_list = create_env(prompt)
            fixed_list = adjust_list(env_list)
            env_list = adjust_overlapping_boxes(fixed_list)
            checked_list = check_columns(env_list)
            json_env = create_json_file(env_list)
        except:
            checked_list = False
    return json_env


def adjust_overlapping_boxes(grid):
    """
    Adjust the grid so that if bounding boxes of objects overlap, the blocks (S or H) in the overlapping
    area of the object with the right-side bounding box are replaced with '-'.
    """
    rows = len(grid)
    cols = len(grid[0])

    def is_block(r, c):
        return grid[r][c] in {"H", "S"}

    def get_neighbors(r, c):
        """Get valid neighboring blocks."""
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and is_block(nr, nc):
                neighbors.append((nr, nc))
        return neighbors

    visited = set()
    object_details = []

    for r in range(rows):
        for c in range(cols):
            if is_block(r, c) and (r, c) not in visited:
                # Start a new object
                queue = [(r, c)]
                visited.add((r, c))
                min_row, min_col, max_row, max_col = r, c, r, c
                object_blocks = set()

                while queue:
                    cr, cc = queue.pop(0)
                    object_blocks.add((cr, cc))
                    for nr, nc in get_neighbors(cr, cc):
                        if (nr, nc) not in visited:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                            min_row, min_col = min(min_row, nr), min(min_col, nc)
                            max_row, max_col = max(max_row, nr), max(max_col, nc)

                object_details.append(
                    ((min_row, min_col), (max_row, max_col), object_blocks)
                )

    # Sort objects by their top-left corner's x-coordinate
    object_details.sort(key=lambda x: x[0][1])

    # Function to check if two boxes overlap
    def boxes_overlap(box1, box2):
        (tl1, br1) = box1
        (tl2, br2) = box2
        return not (
            br1[1] < tl2[1] or br2[1] < tl1[1] or br1[0] < tl2[0] or br2[0] < tl1[0]
        )

    # Modify the grid based on overlapping bounding boxes
    for i in range(len(object_details)):
        for j in range(i + 1, len(object_details)):
            box1, blocks1 = object_details[i][:2], object_details[i][2]
            box2, blocks2 = object_details[j][:2], object_details[j][2]

            if boxes_overlap(box1, box2):
                overlap_area = set()
                for r in range(
                    max(box1[0][0], box2[0][0]), min(box1[1][0], box2[1][0]) + 1
                ):
                    for c in range(
                        max(box1[0][1], box2[0][1]), min(box1[1][1], box2[1][1]) + 1
                    ):
                        if (r, c) in blocks2:
                            overlap_area.add((r, c))

                # Replace overlapping blocks in the right-side box with '-'
                for r, c in overlap_area:
                    grid[r] = grid[r][:c] + "-" + grid[r][c + 1 :]

    return grid


def recreate_fixed_list(json_env):
    """
    Recreates the fixed_list from the json_env object with inverted y-axis.
    The json_env contains information about different objects, their indices in a grid,
    and their types (represented as 5 for 'H' and 2 for 'S').
    The grid is reconstructed into a list of strings, each representing a row, with an inverted y-axis.
    """
    grid_width = json_env["grid_width"]
    grid_height = json_env["grid_height"]
    objects = json_env["objects"]

    # Initialize the grid with '-' (empty space)
    grid = [["-" for _ in range(grid_width)] for _ in range(grid_height)]

    # Populate the grid with objects, inverting the y-axis
    for obj in objects.values():
        indices = obj["indices"]
        types = obj["types"]

        for index, obj_type in zip(indices, types):
            x = index % grid_width
            y = grid_height - 1 - (index // grid_width)  # Invert the y-axis
            grid[y][x] = "H" if obj_type == 5 else "S"  # 'H' for 5 and 'S' for 2

    # Convert grid rows to strings
    fixed_list = ["".join(row) for row in grid]

    return fixed_list


def create_prompt(prompt):
    model_id = "gpt-4"
    # few shot prompting

    # 50% chance to return the original prompt
    if random.random() < 0.5:
        return prompt

    response = openai.ChatCompletion.create(
        model=model_id,
        messages=[
            {
                "role": "system",
                "content": "The prompt describes an two dimensional environment for reinforcement learning, where the task is to learn to move from left to right. Please return a prompt within 100 characters that describes a slightly more difficult environment than the one described in the sent prompt, using words like stairs, hole, wall, shaped mountain, shaped valley, easy, difficult to describe the shape. The reinforcement learning environment consists only of square hard and soft blocks, and the arrangement of these blocks is described in the prompt. There are no other functional blocks. There are no physical forces in this environment other than gravity. Do not alter the phrase 100*20 size Evolution Gym environment.",
            },
            {
                "role": "user",
                "content": "100*20 size Evolution Gym environment that is simple.",
            },
            {
                "role": "assistant",
                "content": "100*20 size Evolution Gym environment that is simple with some small holes.",
            },
            {
                "role": "user",
                "content": "100*20 size Evolution Gym environment that is simple with some small holes.",
            },
            {
                "role": "assistant",
                "content": "100*20 size Evolution Gym environment that is shaped like mountain.",
            },
            {
                "role": "user",
                "content": "100*20 size Evolution Gym environment that is shaped like mountain.",
            },
            {
                "role": "assistant",
                "content": "00*20 size Evolution Gym environment that is a little difficult.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    mutated_prompt = response["choices"][0]["message"]["content"]
    return mutated_prompt


def find_bounding_boxes(grid):
    """
    Find bounding boxes of connected 'H' and 'S' blocks in the grid.
    A bounding box is represented by the coordinates of its top-left and bottom-right corners.
    """
    rows = len(grid)
    cols = len(grid[0])

    def is_block(r, c):
        return grid[r][c] in {"H", "S"}

    def get_neighbors(r, c):
        """Get valid neighboring blocks."""
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and is_block(nr, nc):
                neighbors.append((nr, nc))
        return neighbors

    visited = set()
    bounding_boxes = []

    for r in range(rows):
        for c in range(cols):
            if is_block(r, c) and (r, c) not in visited:
                # Start a new object
                queue = [(r, c)]
                visited.add((r, c))
                min_row, min_col, max_row, max_col = r, c, r, c

                while queue:
                    cr, cc = queue.pop(0)
                    for nr, nc in get_neighbors(cr, cc):
                        if (nr, nc) not in visited:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                            min_row, min_col = min(min_row, nr), min(min_col, nc)
                            max_row, max_col = max(max_row, nr), max(max_col, nc)

                bounding_boxes.append(((min_row, min_col), (max_row, max_col)))

    return bounding_boxes


# Function to print the bounding box area from the grid
def print_bounding_box(grid, box):
    top_left, bottom_right = box
    for row in range(top_left[0], bottom_right[0] + 1):
        print(grid[row][top_left[1] : bottom_right[1] + 1])
