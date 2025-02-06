# Define a function to truncate the array in a file
def truncate_array(file_path, output_path, max_elements=10000):
    """
    Truncates the array in the file to the first `max_elements` and saves it to a new file.

    Parameters:
    - file_path: Path to the input file containing the array.
    - output_path: Path to save the truncated file.
    - max_elements: Maximum number of elements to retain in the array.

    Returns:
    - None
    """
    with open(file_path, "r") as file:
        array_data = file.read()
    
    # Extract elements from the array string
    array_elements = array_data.strip('[]').split(',')
    
    # Truncate to the first `max_elements`
    truncated_array = array_elements[:max_elements]
    
    # Reconstruct the array as a string
    truncated_array_str = '[' + ','.join(truncated_array) + ']'
    
    # Write the truncated array to the output file
    with open(output_path, "w") as output_file:
        output_file.write(truncated_array_str)

# Define file paths
x_file_path = "../data/x.txt"
y_file_path = "../data/y.txt"
z_file_path = "../data/z.txt"

x_output_path = "../data/x_truncated.txt"
y_output_path = "../data/y_truncated.txt"
z_output_path = "../data/z_truncated.txt"

# Truncate the arrays
truncate_array(x_file_path, x_output_path, max_elements=5000)
truncate_array(y_file_path, y_output_path, max_elements=5000)
truncate_array(z_file_path, z_output_path, max_elements=5000)

x_output_path, y_output_path, z_output_path
