# get current path:
import os
def to_output(suffix):
      # Get the current path
      current_path = os.getcwd()

      # Get the parent directory
      parent_dir = os.path.dirname(current_path)

      # Construct the path to the sibling folder "output"
      sibling_folder_path = os.path.join(parent_dir, 'output', suffix)
      
      return sibling_folder_path
